import os
import re
from pathlib import Path
from typing import Literal, TypedDict

import unidic
from fugashi import GenericTagger, Tagger  # type: ignore

import jaconv
from yomikata.dbert import dBert

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.mora_list import MORA_KATA_TO_MORA_PHONEMES, VOWELS
from style_bert_vits2.nlp.japanese.normalizer import replace_punctuation
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


def g2p(
    norm_text: str,
    use_jp_extra: bool = True,
    raise_yomi_error: bool = False,
    use_unidic3: bool = False,
    use_yomikata: bool = False,
    hougen_mode: list[Literal[
        "kinki",
        "kyusyu",
        "convert2b2v",
        "convert2t2ts",
        "convert2d2r",
        "convert2r2d",
        "convert2s2z_sh2j",
        "1st_mora_tyouon",
        "1st_mora_sokuon",
        "1st_mora_remove",
        "1st_mora_renboin",
        "last_mora_acc_h",
        "last_word_acc_1",
        "add_youon_a",
        "add_youon_i",
        "add_youon_e",
        "add_youon_o",
        "hatuonbin",
        "youjigo_like",
    ]] | None = None,
    fugashi_dict: Path | None = None,
    fugashi_user_dict: Path | None = None,
) -> tuple[list[str], list[int], list[int]]:
    """
    他で使われるメインの関数。`normalize_text()` で正規化された `norm_text` を受け取り、
    - phones: 音素のリスト（ただし `!` や `,` や `.` など punctuation が含まれうる）
    - tones: アクセントのリスト、0（低）と1（高）からなり、phones と同じ長さ
    - word2ph: 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
    のタプルを返す。
    ただし `phones` と `tones` の最初と終わりに `_` が入り、応じて `word2ph` の最初と最後に 1 が追加される。

    Args:
        norm_text (str): 正規化されたテキスト
        use_jp_extra (bool, optional): False の場合、「ん」の音素を「N」ではなく「n」とする。Defaults to True.
        raise_yomi_error (bool, optional): False の場合、読めない文字が「'」として発音される。Defaults to False.
        use_unidic3 (bool, optional): True の場合、fugashiとunidic3.10を使用する。Defaults to False.
        hougen_mode (Literal["tokyo", "kinki", "kyusyu"]): use_unidic3がTrueである必要がある。有効値は"tokyo","kinki","kyusyu"。"kinki"の場合アクセントも京阪式になる。b2vはmora bをv に変換し外国語風の訛を作る。Defaults to "tokyo".

    Returns:
        tuple[list[str], list[int], list[int]]: 音素のリスト、アクセントのリスト、word2ph のリスト
    """

    # pyopenjtalk のフルコンテキストラベルを使ってアクセントを取り出すと、punctuation の位置が消えてしまい情報が失われてしまう：
    # 「こんにちは、世界。」と「こんにちは！世界。」と「こんにちは！！！？？？世界……。」は全て同じになる。
    # よって、まず punctuation 無しの音素とアクセントのリストを作り、
    # それとは別に pyopenjtalk.run_frontend() で得られる音素リスト（こちらは punctuation が保持される）を使い、
    # アクセント割当をしなおすことによって punctuation を含めた音素とアクセントのリストを作る。

    # punctuation がすべて消えた、音素とアクセントのタプルのリスト（「ん」は「N」）
    phone_tone_list_wo_punct = __g2phone_tone_wo_punct(norm_text)

    # sep_text: 単語単位の単語のリスト
    # sep_kata: 単語単位の単語のカタカナ読みのリスト、読めない文字は raise_yomi_error=True なら例外、False なら読めない文字を「'」として返ってくる
    sep_text, sep_kata = text_to_sep_kata(norm_text, raise_yomi_error=raise_yomi_error)

    # sep_phonemes: 各単語ごとの音素のリストのリスト
    sep_phonemes = __handle_long([__kata_to_phoneme_list(i) for i in sep_kata])

    # phone_w_punct: sep_phonemes を結合した、punctuation を元のまま保持した音素列
    phone_w_punct: list[str] = []
    for i in sep_phonemes:
        phone_w_punct += i

    # punctuation 無しのアクセント情報を使って、punctuation を含めたアクセント情報を作る
    phone_tone_list = __align_tones(phone_w_punct, phone_tone_list_wo_punct)

    # fugashiで解析
    if use_unidic3 == True:
        sep_text, sep_kata, sep_phonemes, phone_tone_list = update_yomi(
            sep_text,
            sep_kata,
            sep_phonemes,
            phone_w_punct,
            phone_tone_list,
            use_yomiktata= use_yomikata,
            hougen_mode=hougen_mode,
            fugashi_dict=fugashi_dict,
            fugashi_user_dict=fugashi_user_dict,
        )

    # logger.debug(f"phone_tone_list:\n{phone_tone_list}")

    # word2ph は厳密な解答は不可能なので（「今日」「眼鏡」等の熟字訓が存在）、
    # Bert-VITS2 では、単語単位の分割を使って、単語の文字ごとにだいたい均等に音素を分配する

    # sep_text から、各単語を1文字1文字分割して、文字のリスト（のリスト）を作る
    sep_tokenized: list[list[str]] = []
    for i in sep_text:
        if i not in PUNCTUATIONS:
            sep_tokenized.append(
                bert_models.load_tokenizer(Languages.JP).tokenize(i)
            )  # ここでおそらく`i`が文字単位に分割される
        else:
            sep_tokenized.append([i])

    # 各単語について、音素の数と文字の数を比較して、均等っぽく分配する
    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)
        word2ph += __distribute_phone(phone_len, word_len)

    # 最初と最後に `_` 記号を追加、アクセントは 0（低）、word2ph もそれに合わせて追加
    phone_tone_list = [("_", 0)] + phone_tone_list + [("_", 0)]
    word2ph = [1] + word2ph + [1]

    phones = [phone for phone, _ in phone_tone_list]
    tones = [tone for _, tone in phone_tone_list]

    assert len(phones) == sum(word2ph), f"{len(phones)} != {sum(word2ph)}"

    # use_jp_extra でない場合は「N」を「n」に変換
    if not use_jp_extra:
        phones = [phone if phone != "N" else "n" for phone in phones]

    return phones, tones, word2ph


def text_to_sep_kata(norm_text: str, raise_yomi_error: bool = False) -> tuple[list[str], list[str]]:
    """
    `normalize_text` で正規化済みの `norm_text` を受け取り、それを単語分割し、
    分割された単語リストとその読み（カタカナ or 記号1文字）のリストのタプルを返す。
    単語分割結果は、`g2p()` の `word2ph` で1文字あたりに割り振る音素記号の数を決めるために使う。
    例:
    `私はそう思う!って感じ?` →
    ["私", "は", "そう", "思う", "!", "って", "感じ", "?"], ["ワタシ", "ワ", "ソー", "オモウ", "!", "ッテ", "カンジ", "?"]

    Args:
        norm_text (str): 正規化されたテキスト
        raise_yomi_error (bool, optional): False の場合、読めない文字が「'」として発音される。Defaults to False.

    Returns:
        tuple[list[str], list[str]]: 分割された単語リストと、その読み（カタカナ or 記号1文字）のリスト
    """

    # parsed: OpenJTalkの解析結果
    parsed = pyopenjtalk.run_frontend(norm_text)
    sep_text: list[str] = []
    sep_kata: list[str] = []

    for parts in parsed:
        # word: 実際の単語の文字列
        # yomi: その読み、但し無声化サインの`’`は除去
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace("’", "")
        """
        ここで `yomi` の取りうる値は以下の通りのはず。
        - `word` が通常単語 → 通常の読み（カタカナ）
            （カタカナからなり、長音記号も含みうる、`アー` 等）
        - `word` が `ー` から始まる → `ーラー` や `ーーー` など
        - `word` が句読点や空白等 → `、`
        - `word` が punctuation の繰り返し → 全角にしたもの
        基本的に punctuation は1文字ずつ分かれるが、何故かある程度連続すると1つにまとまる。
        他にも `word` が読めないキリル文字アラビア文字等が来ると `、` になるが、正規化でこの場合は起きないはず。
        また元のコードでは `yomi` が空白の場合の処理があったが、これは起きないはず。
        処理すべきは `yomi` が `、` の場合のみのはず。
        """
        assert yomi != "", f"Empty yomi: {word}"
        if yomi == "、":
            # word は正規化されているので、`.`, `,`, `!`, `'`, `-`, `--` のいずれか
            if not set(word).issubset(set(PUNCTUATIONS)):  # 記号繰り返しか判定
                # ここは pyopenjtalk が読めない文字等のときに起こる
                ## 例外を送出する場合
                if raise_yomi_error:
                    raise YomiError(f"Cannot read: {word} in:\n{norm_text}")
                ## 例外を送出しない場合
                ## 読めない文字は「'」として扱う
                logger.warning(f'Cannot read: {word} in:\n{norm_text}, replaced with "\'"')
                # word の文字数分「'」を追加
                yomi = "'" * len(word)
            else:
                # yomi は元の記号のままに変更
                yomi = word
        elif yomi == "？":
            assert word == "?", f"yomi `？` comes from: {word}"
            yomi = "?"
        sep_text.append(word)
        sep_kata.append(yomi)

    return sep_text, sep_kata


def update_yomi(
    sep_text: list[str],
    sep_kata: list[str],
    sep_phonemes: list[list[str]],
    phone_w_punct: list[str],
    phone_tone_list: list[tuple[str, int]],
    use_yomiktata: bool = False,
    hougen_mode: list[Literal[
        "kinki",
        "kyusyu",
        "convert2b2v",
        "convert2t2ts",
        "convert2d2r",
        "convert2r2d",
        "convert2s2z_sh2j",
        "1st_mora_tyouon",
        "1st_mora_sokuon",
        "1st_mora_remove",
        "1st_mora_renboin",
        "last_mora_acc_h",
        "last_word_acc_1",
        "add_youon_a",
        "add_youon_i",
        "add_youon_e",
        "add_youon_o",
        "hatuonbin",
        "youjigo_like",
    ]] | None = None,
    fugashi_dict: Path | None = None,
    fugashi_user_dict: Path | None = None,
) -> tuple[
    list[str],
    list[str],
    list[list[str]],
    list[tuple[str, int]],
]:
    """
    fugashiで比較的新しいunidicを使って読みを取得し、openjtalkの古い読みと比較して一致していない場合更新する。
    アクセント取得時にopenjtalkでもう一回処理を通す時のために、norm_textの該当箇所をカタカナに変更する。

    Args:
        sep_text (list[str]):
        sep_kata (list[str]):
        phone_tone_list (list[tuple[str, int]]):

    Returns:
        tuple[list[str], list[str],  list[list[str]], list[tuple[str, int]], ]:
    """
    # fugashiで使用するunidicは別ライブラリに分けられている
    # fygashiで使用するuniduicのバージョンは3.10
    # unidic-liteは2.13なのでたぶんopenjtalk同等

    # openjtalkの辞書のバージョンについて
    # １．アクセント情報のないipadicにアクセント情報を足したもの
    # ipadicは2011年で更新が止まっているがどのバージョンを使用しているのかわからなかった
    # ２，上記の辞書と情報の配列が同じunidic
    # openjtalkの更新時期(2018)から推測してunidic-csj-2.3.0以前

    word_list: list[str] = []
    kana_list: list[str] = []
    accent_list: list[str] = []
    pos_list: list[str] = []

    for text in sep_text:
        # 現在のテキストがpyopenjtalkのユーザー辞書にない場合
        if type(text) == str:
            cur_word_list, cur_kana_list, cur_accent_list, cur_pos_list = __fugashi_sep_kata(
                text, dict_path=fugashi_dict, user_dict_path=fugashi_user_dict
            )  # user_dict_path=compiled_dict_path

            word_list += cur_word_list
            kana_list += cur_kana_list
            accent_list += cur_accent_list  # type: ignore
            pos_list += cur_pos_list

    # 同音異義語処理
    if use_yomiktata == True:
        kana_list = __yomikata_patch(word_list, kana_list)

    # 方言処理
    if hougen_mode != None:
        kana_list, accent_list = __hougen_patch(kana_list, accent_list, pos_list, hougen_mode)

    # 京阪式アクセント処理
    if hougen_mode == "kinki":
        accent_list = __keihan_patch(kana_list, accent_list, pos_list)

    new_sep_text: list[str] = word_list
    new_sep_kata: list[str] = kana_list
    accent_hl_list: list[str] = []

    for num1 in range(len(kana_list)):
        cur_kana = kana_list[num1]
        cur_accent = accent_list[num1]

        cur_acc_hl = __convert_acc2hl(cur_kana, cur_accent)
        accent_hl_list += cur_acc_hl

    # new_sep_phonemes: 各単語ごとの音素のリストのリスト
    new_sep_phonemes = __handle_long([__kata_to_phoneme_list(i) for i in kana_list])

    # phone_w_punct: new_sep_phonemes を結合した音素列
    new_phone_w_punct: list[str] = []
    for i in new_sep_phonemes:
        new_phone_w_punct += i

    # 音素数とアクセント数が一致しない場合
    assert len(accent_hl_list) == len(
        new_phone_w_punct
    ), f"accent list num:{len(accent_hl_list)} != phone_list num:{len(new_phone_w_punct)}"

    # new_phone_tone_list == 新しいphone_tone_list(そのままの意味)
    new_phone_tone_list = []

    for i in range(len(new_phone_w_punct)):
        phone = new_phone_w_punct[i]
        accent = accent_hl_list[i]
        new_phone_tone_list.append([phone, int(accent)])

    # 標準語の場合
    if hougen_mode == "tokyo":
        # 音素が完全一致して区切った数も一致した場合openjtalkの出力したアクセントを使う。
        if phone_w_punct == new_phone_w_punct and len(kana_list) == len(sep_kata):
            return sep_text, sep_kata, sep_phonemes, phone_tone_list

    # そうでない場合は区切り方が間違っているので新しいものを使う
    return new_sep_text, new_sep_kata, new_sep_phonemes, new_phone_tone_list


def __fugashi_sep_kata(
    text: str, dict_path: Path | None = None, user_dict_path: Path | None = None
) -> tuple[list[str], list[str], list[str | list[str]], list[str]]:
    # ユーザー辞書がある場合
    if user_dict_path != None:
        # ユーザー辞書のディレクトリ
        user_dict_path_str: str = str(user_dict_path)

        # 辞書が指定されていない場合はデフォルトの辞書を使う
        if dict_path == None:
            dict_path_str = unidic.DICDIR
            dicrc_path = Path(dict_path_str) / "dicrc"
        else:
            dict_path_str = str(dict_path)
            dicrc_path = dict_path / "dicrc"

        # windows環境だと\が途中でエスケープされるバグがあるので二重にする
        if os.name == "nt":
            user_dict_path_str: str = str(user_dict_path).replace("\\", "\\\\")
            dict_path_str: str = dict_path_str.replace("\\", "\\\\")
            dicrc_path_str: str = str(dicrc_path).replace("\\", "\\\\")
            # ユーザー辞書を読ませる場合は辞書も引数で読ませる必要がある
            tagger = GenericTagger(f"-r {dicrc_path_str} -Owakati -d {dict_path_str} -u {user_dict_path_str}")
        else:
            tagger = GenericTagger(f"-r {dicrc_path!s} -Owakati -d {dict_path_str!s} -u {user_dict_path!s}")

    # 辞書が指定されている場合
    elif dict_path != None:
        dicrc_path = dict_path / "dicrc"
        # windows環境だと\が途中でエスケープされるバグがあるので二重にする
        if os.name == "nt":
            dict_path_str: str = str(dict_path).replace("\\", "\\\\")
            dicrc_path_str: str = str(dicrc_path).replace("\\", "\\\\")
            tagger = GenericTagger(f"-r {dicrc_path_str} -Owakati -d {dict_path_str}")
        else:
            tagger = GenericTagger(f"-r {dicrc_path!s} -Owakati -d {dict_path!s}")
    else:
        tagger = Tagger("-Owakati")

    word_list: list[str] = []
    kana_list: list[str] = []
    accent_list: list[str | list[str]] = []
    pos_list: list[str] = []

    # 解析
    for word in tagger(text):
        feature = word.feature_raw

        # アクセント核が二つある場合,"*,*",という風に記述されているので,を/に変更し"を消す
        if re.search(r"\".*,.*\"", feature):
            accStart = re.search(r"\".*,.*\"", feature).start()  # type: ignore
            accEnd = re.search(r"\".*,.*\"", feature).end()  # type: ignore

            accent = feature[accStart:accEnd].replace(",", "/")

            feature = feature[:accStart] + accent.replace('"', "") + feature[accEnd:]

        feature = feature.split(",")

        # "feature" is the Unidic feature data as a named tuple
        # feature_rawはその語句の生の特徴情報

        # 得られる分類情報についてのメモ

        # unidicの
        # 0から数えて0番目(csv形式の時0から数えて4番目)が分類一
        # 0から数えて9番目(csv形式の時0から数えて13番目)が発音系
        # 0から数えて24番目(csv形式の時0から数えて28番目)がアクセントタイプ
        # 0から数えて25番目(csv形式の時0から数えて29番目)がアクセント結合型

        # 辞書にある場合
        if len(feature) == 29:
            pos1: str = feature[0]
            kana: str = feature[9]
            accent = feature[24]

            # 読みがない場合
            if kana == "*":
                kana = "'"

            # 感嘆符か疑問符の場合
            if re.match(r"[!?]+", str(word)):
                kana = str(word)

            word_list.append(str(word))
            kana_list.append(kana)
            accent_list.append(accent)
            pos_list.append(pos1)

        # 辞書にない場合の処理
        elif re.match(r"[ァ-ロワ-ヴぁ-ろわ-ん－a-zA-Zａ-ｚＡ-Ｚ]+", str(word)):
            word, kana = text_to_sep_kata(str(word), raise_yomi_error=False)  # type: ignore
            word_list += word
            kana_list += kana

            # Xbox等fugashi解析時につながっていてもpyopenjtlkでは x box に分かれる
            # なので分かれた数だけアクセントを追加する
            for i in range(len(word)):
                accent_list.append("0")
                pos_list.append("未分類")

        else:
            kana = "'"
            accent = "*"
            pos1 = "未分類"

            word_list.append(str(word))
            kana_list.append(kana)
            accent_list.append(accent)
            pos_list.append(pos1)

    # fugashiでアクセント未取得時、0に設定
    for i in range(len(accent_list)):
        if str(accent_list[i]) == "*":
            accent_list[i] = "0"

        # アクセントの種類が2つ以上の場合先頭のものを使う
        elif len(accent_list[i]) == 3:
            accent_list[i] = str(accent_list[i]).split("/")[0]

    return word_list, kana_list, accent_list, pos_list


def __convert_acc2hl(
    kana: str,
    accent: str,  # type: ignore
) -> list[str]:
    # アクセントの変換処理
    # カタカナの読みの文字数(= 拍数 = mora数)から音素数のアクセント配列のリストに変換
    # 後で音素と合成する

    # toneのリスト
    accent_hl_list: list[str] = []

    # 拗音の正規表現
    _YOUON_PATTERN = re.compile(r"[ァィゥェォャュョヮ]")

    mora = len(kana)

    # すべて1にする
    if accent == "ALL_H":
        for phone in __kata_to_phoneme_list(kana):
            accent_hl_list.append("1")
        return accent_hl_list
    # すべて0にする
    elif accent == "ALL_L":
        for phone in __kata_to_phoneme_list(kana):
            accent_hl_list.append("0")
        return accent_hl_list

    accent: int = int(accent)

    # 一文字の場合
    if mora == 1:
        if accent == 1:
            # 返した音素のリストの数だけ実行
            for phone in __kata_to_phoneme_list(kana):
                accent_hl_list.append("1")

        else:
            for phone in __kata_to_phoneme_list(kana):
                accent_hl_list.append("0")

    # 二文字で拗音が続く場合
    elif mora == 2 and _YOUON_PATTERN.fullmatch(kana[1]):
        if accent == 1:
            for phone in __kata_to_phoneme_list(kana):
                accent_hl_list.append("1")

        else:
            for phone in __kata_to_phoneme_list(kana):
                accent_hl_list.append("0")

    # アクセント核が平型の場合平型に
    elif accent == 0:
        # 2文字目に拗音が続く場合
        # 例　"キャ" など
        if _YOUON_PATTERN.fullmatch(kana[1]):
            # 先頭を追加
            for phone in __kata_to_phoneme_list(kana[:2]):
                accent_hl_list.append("0")

            for phone in __kata_to_phoneme_list(kana[2:]):
                accent_hl_list.append("1")

        # 拗音が続かない場合
        else:
            # 先頭を追加
            for phone in __kata_to_phoneme_list(kana[0]):
                accent_hl_list.append("0")

            for phone in __kata_to_phoneme_list(kana[1:]):
                accent_hl_list.append("1")

    # アクセント核が先頭の場合先頭を先に音素に変換しアクセントを設定する。
    # 例　"ア"  =>　"a" -> ("a", "1")
    elif accent == 1:
        # 2文字目に拗音が続く場合
        # 例　"キャ" など
        if _YOUON_PATTERN.fullmatch(kana[1]):
            # 2文字目までを追加
            for phone in __kata_to_phoneme_list(kana[:2]):
                accent_hl_list.append("1")

            # 3文字目以降を追加
            for phone in __kata_to_phoneme_list(kana[2:]):
                accent_hl_list.append("0")

        # 拗音が続かない場合
        else:
            # 1文字目を追加
            for phone in __kata_to_phoneme_list(kana[0]):
                accent_hl_list.append("1")

            # 2文字目以降を追加
            for phone in __kata_to_phoneme_list(kana[1:]):
                accent_hl_list.append("0")

    # アクセント核が先端と終端の間に位置する場合先頭を先に音素に変換しアクセントを設定する。
    elif accent < mora:
        # acc = 0スタートに直した 泊で数えた アクセント数
        acc = accent - 1

        # 2文字目に拗音が続く場合でアクセント核が3文字目の場合
        # 例　"フェニックス" など
        if _YOUON_PATTERN.fullmatch(kana[1]) and accent == 2:
            # 2文字目までを追加
            for phone in __kata_to_phoneme_list(kana[:2]):
                accent_hl_list.append("0")

            # 3文字目以降を追加
            for phone in __kata_to_phoneme_list(kana[2]):
                accent_hl_list.append("1")

            for phone in __kata_to_phoneme_list(kana[3:]):
                accent_hl_list.append("0")

        # アクセント核の2文字目に拗音が続く場合
        # 例　"インフェルノ" など
        elif _YOUON_PATTERN.fullmatch(kana[acc + 1]):
            # アクセント核までを追加
            for phone in __kata_to_phoneme_list(kana[:acc]):
                accent_hl_list.append("0")

            # アクセント核以降を追加
            for phone in __kata_to_phoneme_list(kana[acc : accent + 1]):
                accent_hl_list.append("1")

            for phone in __kata_to_phoneme_list(kana[accent + 1 :]):
                accent_hl_list.append("0")

        else:
            # 先頭からアクセント核まで
            for phone in __kata_to_phoneme_list(kana[:acc]):
                accent_hl_list.append("0")

            # アクセント核を追加
            for phone in __kata_to_phoneme_list(kana[acc]):
                accent_hl_list.append("1")

            # アクセント核の一個先から終端まで
            for phone in __kata_to_phoneme_list(kana[accent:]):
                accent_hl_list.append("0")

    # アクセント核が終端の場合先頭を先に音素に変換しアクセントを設定する。
    elif accent == mora:
        # acc = 0スタートに直した 泊で数えた アクセント数
        acc = accent - 1

        for phone in __kata_to_phoneme_list(kana[:acc]):
            accent_hl_list.append("0")

        # 終端を追加
        for phone in __kata_to_phoneme_list(kana[acc]):
            accent_hl_list.append("1")

    return accent_hl_list

__YOMI_PATTERN = re.compile(r"..*/..*")

def __yomikata_patch(sep_text: list[str],sep_kata:list[str]) -> list[str]:
    norm_text = "".join(sep_text)

    reader = dBert()
    out_text:str = reader.furigana(norm_text)

    # 読みを降ったほうがいい文字が {何/なに}が{何/なん}でも のような形式になる

    #{}/|は正規化で消えるので。textに混ざることはない。正規化を変更したときはこの処理も変える必要がある。
    out_text = out_text.replace("{","|")
    out_text = out_text.replace("}","|")
    out_list = out_text.split("|")

    convert_list_text:list[str] = []
    convert_list_kana:list[str] = []

    for i in out_list:
        if __YOMI_PATTERN.fullmatch(i):

            word = i.split("/")
            convert_list_text.append(word[0])
            # 読みをカタカナからひらがなに変更
            kana = jaconv.hira2kata(word[1])
            convert_list_kana.append(kana)

    for i in range(0, len(sep_text)):
        for ii in range(0, len(convert_list_text)):
            # 読みを上書きすべき文字があったら
            if sep_text[i] == convert_list_text[ii]:
                sep_kata[i] = convert_list_kana[ii]

    return sep_kata

__KYUSYU_HATUON_PATTERN = re.compile("[ヌニムモミ]+")
__YOUON_PATTERN = re.compile("[ァィゥェォャュョヮ]+")
__A_DAN_PATTERN = re.compile("[アカサタナハマヤラワガダバパ]|[ャヮ]+")
__I_DAN_PATTERN = re.compile("[イキシチニヒミリギジビピ]|ィ+")
__E_DAN_PATTERN = re.compile("[エケセテネヘメレゲデベペ]|ェ+")
__O_DAN_PATTERN = re.compile("[オコソトノホモヨロゴゾドボポ]|[ョォ]+")


def __hougen_patch(
    sep_kata: list[str],
    sep_acc: list[str],
    sep_pos: list[str],
    hougen_id: list[Literal[
        "kinki",
        "kyusyu",
        "convert2b2v",
        "convert2t2ts",
        "convert2d2r",
        "convert2r2d",
        "convert2s2z_sh2j",
        "1st_mora_tyouon",
        "1st_mora_sokuon",
        "1st_mora_remove",
        "1st_mora_renboin",
        "last_mora_acc_h",
        "last_word_acc_1",
        "add_youon_a",
        "add_youon_i",
        "add_youon_e",
        "add_youon_o",
        "hatuonbin",
        "youjigo_like",
    ]]
) -> tuple[list[str], list[str]]:
    """
    NHK日本語アクセント辞典を参考に方言の修正を加える。
    区分は付録NHK日本語アクセント辞典125ｐを参照した。
    持っていない人のためにも、細かくコメントを残しておく。

    Args:
        sep_kata (list[str]):
        sep_kata (list[str]):
        hougen_id (str):
    Returns:
        sep_kata(list[str]): 修正された sep_kata
    """

    # 区分は以下の通り
    # 本土方言/
    #   八丈方言
    #   東部方言
    #   西部方言 /
    #         近畿方言 kinki
    #   九州方言 kyusyu/
    #

    # 以下厳密でない方言もしくは喋り方

    #   bをvに変換する convert2b2v
    #   bをvに変換する convert2t2ts
    #   dをrに変換し、アクセントを頭高型にする convert2d2r

    #   文章の１モーラ目を長音化しアクセントを頭高型に 1st_mora_tyouon ;　やはり、　＝＞　やーはり HLLL
    #   文章の１モーラ目を撥音化しアクセントを頭高型に 1st_mora_sokuon ;　やはり、　＝＞ やっはり　HLLL
    #   文章の１モーラ目をしアクセントを平型に っ　に変換 1st_mora_remove ;　やはり、　＝＞　っはり　LHH

    #   最後の単語の終端をアクセント核にする last_mora_acc_h
    #   最後のアクセントを頭高型に last_word_acc_1

    #   単語の先頭以外ののno,ra,ruをNに変換する hatuonbin

    #   sをchに変換する　youjigo_like ;(幼児語のネイティブ話者つまり幼児の喋る幼児語でなく、我々大人の喋る(イメージする)幼児語である)

    #   各単語に最初にあ行がでてきた時"ァ"をつけ"ァ"をアクセント核にする。add_youon_a ;　そうさ、ボクの仕業さ。悪く思うなよ　＝＞　そうさぁ。ボクの仕業さぁ。わぁるく思うなぁよ
    #   ァはアに置き換えられるのでーでもアでもよいが、わかりやすくするためァとした。
    #   各単語に最初にい行がでてきた時"ィ"をつけアクセントを頭高型にする。add_youon_i ;　しまった。にげられた。　＝＞　しぃまった。にぃげられた
    #   各単語に最初にえ行がでてきた時"ェ"をつけ"ェ"をアクセント核にする。add_youon_e ;　へえ、それで　＝＞　へェえ、それェでェ
    #   各単語に最初にお行がでてきた時"ぉ"をつけアクセントを頭高型にする。add_youon_i ;　ようこそ。　＝＞　よぉうこぉそ。
    #   各単語に最初を連母音にしアクセント頭高型にする。e は ei o は ou になる。1st_mora_renboin ;　俺のターン。　＝＞　おぅれのターン。 , 先生。　せぃんせい


    if "kyusyu" in hougen_id:
        for i in range(len(sep_kata)):
            # 九州のほぼ全域で e を ye と発音する；付録131ｐ
            sep_kata[i] = sep_kata[i].replace("エ", "イェ")

            # 九州のほぼ全域で s eをsh e , z eをj eと発音する；付録132ｐ
            sep_kata[i] = sep_kata[i].replace("セ", "シェ")
            sep_kata[i] = sep_kata[i].replace("ゼ", "ジェ")

            # 発音化：語末のヌ、ニ、ム、モ、ミなどが発音 ンN で表される。；付録132ｐ
            num = len(sep_kata[i])
            if __KYUSYU_HATUON_PATTERN.fullmatch(sep_kata[i][num - 1]):
                sep_kata[i] = sep_kata[i][: num - 1] + "ン"

    if "kinki" in hougen_id:
        for i in range(len(sep_kata)):
            # 1泊の名詞を長音化し2泊で発音する
            if sep_pos[i] == "名詞" and len(sep_kata[i]) == 1:
                if sep_kata[i] == "!" or "?" or "'":
                    sep_kata[i] = sep_kata[i] + "ー"

    # ここから特に参考資料はないが表現の幅が広がったり、話者の特性を再現できそうなもの
    if "convert2b2v" in hougen_id:
        for i in range(len(sep_kata)):
            if "バ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("バ", "ヴァ")
            if "ビ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ビ", "ヴィ")
            if "ブ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ブ", "ヴ")
            if "ベ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ベ", "ヴェ")
            if "ボ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ボ", "ヴォ")

    if "convert2t2ts" in hougen_id:
        for i in range(len(sep_kata)):
            if "タ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("タ", "ツァ")
            if "チ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("チ", "ツィ")
            if "テ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("テ", "ツェ")
            if "ト" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ト", "ツォ")

    if "convert2d2r" in hougen_id:
        for i in range(len(sep_kata)):
            if "ダ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ダ", "ラ")

            if "デ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("デ", "レ")

            if "ド" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ド", "ロ")

            # アクセントを平型に変更
            sep_acc[0] = "0"

    if "convert2r2d" in hougen_id:
        for i in range(len(sep_kata)):
            if "ラ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ラ", "ダ")

            if "レ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("レ", "デ")

            if "ロ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ロ", "ド")

            # アクセントを頭高型に変更
            sep_acc[0] = "1"

    if "convert2s2z_sh2j" in hougen_id:
        for i in range(len(sep_kata)):
            if "サ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("サ", "ザ")

            if "スィ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("スィ", "ズィ")

            if "ス" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ス", "ズ")

            if "セ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("セ", "ゼ")

            if "ソ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ソ", "ゾ")

            if "シャ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("シャ", "ジャ")

            if "シ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("シ", "ジ")

            if "シュ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("シュ", "ジュ")

            if "シェ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("シェ", "ジェ")

            if "ショ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ショ", "ジョ")

            # アクセントを頭高型に変更
            sep_acc[0] = "1"


    if "hatuonbin" in hougen_id:
        for i in range(len(sep_kata)):
            #1文字以外の時
            if len(str(sep_kata[i])) != 1:
                # 各単語先頭と終端は置き換えない
                if "ナ" in str(sep_kata[i][1:-1]):
                    sep_kata[i] = sep_kata[i].replace("ナ", "ン")
                elif "ノ" in str(sep_kata[i][1:-1]):
                    sep_kata[i] = sep_kata[i].replace("ノ", "ン")
                # 一種ずつしか撥音化しない
                elif "ル" in str(sep_kata[i][1:-1]):
                    sep_kata[i] = sep_kata[i].replace("ル", "ン")
                elif "ラ" in str(sep_kata[i][1:-1]):
                    sep_kata[i] = sep_kata[i].replace("ラ", "ン")

    if "youjigo_like" in hougen_id:
        for i in range(len(sep_kata)):
            if "サ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("サ", "チャ")
            if "シ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("シ", "チ")
            if "ス" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ス", "チュ")
            if "セ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("セ", "チェ")
            if "ソ" in str(sep_kata[i]):
                sep_kata[i] = sep_kata[i].replace("ソ", "チョ")

    if "add_youon_a" in hougen_id:
        for i in range(len(sep_kata)):
            pos = __A_DAN_PATTERN.search(str(sep_kata[i]))

            if pos:
                sep_kata[i] = sep_kata[i][: pos.end()] + "ァ" + sep_kata[i][pos.end() :]  # type:ignore

                if type(pos.end()) == int:
                        # ァがアクセント核になる
                        sep_acc[i] = str(pos.end())  # type:ignore


    if "add_youon_i" in hougen_id:
        for i in range(len(sep_kata)):
            pos = __I_DAN_PATTERN.search(str(sep_kata[i]))

            if pos:
                # マッチした語が最後の以外で　シャ　等　マッチした文字の後に拗音が来ない場合
                if len(str(sep_kata[i])) > pos.end() and sep_kata[i][pos.end()] != "ャ":# type:ignore
                    sep_kata[i] = sep_kata[i][: pos.end()] + "ィ" + sep_kata[i][pos.end() :]  # type:ignore

                    if type(pos.end()) == int:
                        # アクセントを頭高型にする。
                            sep_acc[i] = "1"  # type:ignore

                # 上記以外のャが入っていない条件
                elif "ャ" not in sep_kata[i]:
                    sep_kata[i] = sep_kata[i][: pos.end()] + "ィ" + sep_kata[i][pos.end() :]  # type:ignore

                    if type(pos.end()) == int:
                            # アクセントを頭高型にする
                            sep_acc[i] = "1"  # type:ignore

    if "add_youon_e" in hougen_id:
        for i in range(len(sep_kata)):
            pos = __E_DAN_PATTERN.search(str(sep_kata[i]))

            if pos:
                sep_kata[i] = sep_kata[i][: pos.end()] + "ェ" + sep_kata[i][pos.end() :]  # type:ignore
                # ェがアクセント核になる
                if type(pos.end()) == int:
                    sep_acc[i] = str(pos.end())  # type:ignore

    if "add_youon_o" in hougen_id:
        for i in range(len(sep_kata)):
            pos = __O_DAN_PATTERN.search(str(sep_kata[i]))

            if pos:
                sep_kata[i] = sep_kata[i][: pos.end()] + "ォ" + sep_kata[i][pos.end() :]  # type:ignore
                # アクセントを頭高型にする。
                if type(pos.end()) == int:
                    sep_acc[i] = "1"  # type:ignore

    if "1st_mora_tyouon" in hougen_id:

        pos = __YOUON_PATTERN.search(str(sep_kata[0]))

        if pos:
            # マッチしたパターンが二文字目から(一文字文字以内の場合)
            if pos.start() == 1:
                sep_kata[0] = sep_kata[0][: pos.end()] + "ー" + sep_kata[0][pos.end() :]  # type:ignore
        else:
            sep_kata[0] = sep_kata[0][0] + "ー" + sep_kata[0][1:]

        # bアクセントを頭高型に変更
        sep_acc[0] = "1"

    if "1st_mora_sokuon" in hougen_id:

        pos = __YOUON_PATTERN.search(str(sep_kata[0]))

        if pos:
            # マッチしたパターンが二文字目から(一文字文字以内の場合)
            if pos.start() == 1:
                sep_kata[0] = sep_kata[0][: pos.end()] + "ッ" + sep_kata[0][pos.end() :]  # type:ignore
        else:
            sep_kata[0] = sep_kata[0][0] + "ッ" + sep_kata[0][1:]

        # bアクセントを頭高型に変更
        sep_acc[0] = "1"

    if "1st_mora_remove" in hougen_id:

        pos = __YOUON_PATTERN.search(str(sep_kata[0]))

        if pos:
            # マッチしたパターンが二文字目からでかつ伸ばす必要がある(一文字文字以内の場合)
            if pos.start() == 1:
                sep_kata[0] = "ッ" + sep_kata[0][pos.end() :]  # type:ignore
        else:
            sep_kata[0] = "ッ" + sep_kata[0][1:]

    if "1st_mora_renboin"  in hougen_id:

        pos = __O_DAN_PATTERN.search(str(sep_kata[0]))

        if pos:
            sep_kata[0] = sep_kata[0][: pos.end()] + "ゥ" + sep_kata[0][pos.end() :]  # type:ignore
            # アクセントを頭高型に
            if type(pos.end()) == int:
                sep_acc[0] = "1"  # type:ignore
        else:
            pos = __E_DAN_PATTERN.search(str(sep_kata[0]))

            if pos:
                sep_kata[0] = sep_kata[0][: pos.end()] + "ィ" + sep_kata[0][pos.end() :]  # type:ignore
                # アクセントを頭高型に
                if type(pos.end()) == int:
                    sep_acc[0] = "1"  # type:ignore

    if "last_mora_acc_h" in hougen_id:
        # 最後の単語の終端をアクセント核にする
        last_word = sep_kata[len(sep_kata)-1]
        sep_acc[len(sep_acc)-1] = str(len(last_word))


    if "last_word_acc_1" in hougen_id:
        # 最後のアクセントを頭高型に
        sep_acc[len(sep_acc)-1] = "1"


    return sep_kata, sep_acc


def __keihan_patch(
    sep_kata: list[str],
    sep_acc: list[str],
    sep_pos: list[str],
) -> list[str]:
    """
    NHK日本語アクセント辞典を参考にアクセントを京阪式にする
    東京式と京阪式の対応表は付録146ｐを参照した
    持っていない人のためにも、細かくコメントを残しておく

    Args:
        sep_kata (list[str]):
        sep_acc (list[str]):
        sep_pos (list[str]):
    Returns:
        sep_acc (list[str]): 修正された sep_acc
    """

    for i in range(len(sep_pos)):
        # 分類が名詞の場合
        if sep_pos[i] == "名詞":
            # 一音の場合(長音可で2泊化)されている
            if sep_kata[i][1] == "ー":
                # 平型の場合頭高型に
                if sep_acc[i] == "0":
                    sep_acc[i] = "1"
                # 頭高型の場合全て低く
                if sep_acc[i] == "ALL_L":
                    sep_acc[i] = "0"
            # ニ音の場合
            elif len(sep_kata[i]) == 2:
                # 平型の場合全て高く
                if sep_acc[i] == "0":
                    sep_acc[i] = "ALL_H"
                # 尾高型の場合頭高型に
                if sep_acc[i] == "2":
                    sep_acc[i] = "1"

        # 分類が動詞の場合
        elif sep_pos[i] == "動詞":
            # ニ音の場合
            if len(sep_kata[i]) == 2:
                # 平型の場合全て高く
                if sep_acc[i] == "0":
                    sep_acc[i] = "ALL_H"
                # 頭高型の場合頭高型に
                if sep_acc[i] == "1":
                    sep_acc[i] = "2"
            # 三音の場合
            if len(sep_kata[i]) == 3:
                # 平型の場合全て高く
                if sep_acc[i] == "0":
                    sep_acc[i] = "ALL_H"
                # 中高型の場合尾高型に
                if sep_acc[i] == "2":
                    sep_acc[i] = "3"

        # 分類が形容詞の場合
        elif sep_pos[i] == "形容詞":
            # ニ音の場合
            if len(sep_kata[i]) == 2:
                # 頭高型の場合頭高型に
                if sep_acc[i] == "1":
                    sep_acc[i] = "2"
            # 三音の場合
            if len(sep_kata[i]) == 3:
                # 平型の場合頭高に
                if sep_acc[i] == "0":
                    sep_acc[i] = "1"
                # 中高型の場合頭高に
                if sep_acc[i] == "2":
                    sep_acc[i] = "1"
    return sep_acc


def adjust_word2ph(
    word2ph: list[int],
    generated_phone: list[str],
    given_phone: list[str],
) -> list[int]:
    """
    `g2p()` で得られた `word2ph` を、generated_phone と given_phone の差分情報を使っていい感じに調整する。
    generated_phone は正規化された読み上げテキストから生成された読みの情報だが、
    given_phone で 同じ読み上げテキストに異なる読みが与えられた場合、正規化された読み上げテキストの各文字に
    音素が何文字割り当てられるかを示す word2ph の合計値が given_phone の長さ (音素数) と一致しなくなりうる
    そこで generated_phone と given_phone の差分を取り変更箇所に対応する word2ph の要素の値だけを増減させ、
    アクセントへの影響を最低限に抑えつつ word2ph の合計値を given_phone の長さ (音素数) に一致させる。

    Args:
        word2ph (list[int]): 単語ごとの音素の数のリスト
        generated_phone (list[str]): 生成された音素のリスト
        given_phone (list[str]): 与えられた音素のリスト

    Returns:
        list[int]: 修正された word2ph のリスト
    """

    # word2ph・generated_phone・given_phone 全ての先頭と末尾にダミー要素が入っているので、処理の都合上それらを削除
    # word2ph は先頭と末尾に 1 が入っている (返す際に再度追加する)
    word2ph = word2ph[1:-1]
    generated_phone = generated_phone[1:-1]
    given_phone = given_phone[1:-1]

    class DiffDetail(TypedDict):
        begin_index: int
        end_index: int
        value: list[str]

    class Diff(TypedDict):
        generated: DiffDetail
        given: DiffDetail

    def extract_differences(generated_phone: list[str], given_phone: list[str]) -> list[Diff]:
        """
        最長共通部分列を基にして、二つのリストの異なる部分を抽出する。
        """

        def longest_common_subsequence(X: list[str], Y: list[str]) -> list[tuple[int, int]]:
            """
            二つのリストの最長共通部分列のインデックスのペアを返す。
            """
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            # LCSの長さを構築
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])
            # LCSを逆方向にトレースしてインデックスのペアを取得
            index_pairs = []
            i, j = m, n
            while i > 0 and j > 0:
                if X[i - 1] == Y[j - 1]:
                    index_pairs.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                elif L[i - 1][j] >= L[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
            index_pairs.reverse()
            return index_pairs

        differences = []
        common_indices = longest_common_subsequence(generated_phone, given_phone)
        prev_x, prev_y = -1, -1

        # 共通部分のインデックスを基にして差分を抽出
        for x, y in common_indices:
            diff_X = {
                "begin_index": prev_x + 1,
                "end_index": x,
                "value": generated_phone[prev_x + 1 : x],
            }
            diff_Y = {
                "begin_index": prev_y + 1,
                "end_index": y,
                "value": given_phone[prev_y + 1 : y],
            }
            if diff_X or diff_Y:
                differences.append({"generated": diff_X, "given": diff_Y})
            prev_x, prev_y = x, y
        # 最後の非共通部分を追加
        if prev_x < len(generated_phone) - 1 or prev_y < len(given_phone) - 1:
            differences.append(
                {
                    "generated": {
                        "begin_index": prev_x + 1,
                        "end_index": len(generated_phone) - 1,
                        "value": generated_phone[prev_x + 1 : len(generated_phone) - 1],
                    },
                    "given": {
                        "begin_index": prev_y + 1,
                        "end_index": len(given_phone) - 1,
                        "value": given_phone[prev_y + 1 : len(given_phone) - 1],
                    },
                }
            )
        # generated.value と given.value の両方が空の要素を diffrences から削除
        for diff in differences[:]:
            if len(diff["generated"]["value"]) == 0 and len(diff["given"]["value"]) == 0:
                differences.remove(diff)

        return differences

    # 二つのリストの差分を抽出
    differences = extract_differences(generated_phone, given_phone)

    # word2ph をもとにして新しく作る word2ph のリスト
    ## 長さは word2ph と同じだが、中身は 0 で初期化されている
    adjusted_word2ph: list[int] = [0] * len(word2ph)
    # 現在処理中の generated_phone のインデックス
    current_generated_index = 0

    # word2ph の要素数 (=正規化された読み上げテキストの文字数) を維持しながら、差分情報を使って word2ph を修正
    ## 音素数が generated_phone と given_phone で異なる場合にこの adjust_word2ph() が呼び出される
    ## word2ph は正規化された読み上げテキストの文字数に対応しているので、要素数はそのまま given_phone で増減した音素数に合わせて各要素の値を増減する
    for word2ph_element_index, word2ph_element in enumerate(word2ph):
        # ここの word2ph_element は、正規化された読み上げテキストの各文字に割り当てられる音素の数を示す
        # 例えば word2ph_element が 2 ならば、その文字には 2 つの音素 (例: "k", "a") が割り当てられる
        # 音素の数だけループを回す
        for _ in range(word2ph_element):
            # difference の中に 処理中の generated_phone から始まる差分があるかどうかを確認
            current_diff: Diff | None = None
            for diff in differences:
                if diff["generated"]["begin_index"] == current_generated_index:
                    current_diff = diff
                    break
            # current_diff が None でない場合、generated_phone から始まる差分がある
            if current_diff is not None:
                # generated から given で変わった音素数の差分を取得 (2増えた場合は +2 だし、2減った場合は -2)
                diff_in_phonemes = \
                    len(current_diff["given"]["value"]) - len(current_diff["generated"]["value"])  # fmt: skip
                # adjusted_word2ph[(読み上げテキストの各文字のインデックス)] に上記差分を反映
                adjusted_word2ph[word2ph_element_index] += diff_in_phonemes
            # adjusted_word2ph[(読み上げテキストの各文字のインデックス)] に処理が完了した分の音素として 1 を加える
            adjusted_word2ph[word2ph_element_index] += 1
            # 処理中の generated_phone のインデックスを進める
            current_generated_index += 1

    # この時点で given_phone の長さと adjusted_word2ph に記録されている音素数の合計が一致しているはず
    assert len(given_phone) == sum(adjusted_word2ph), f"{len(given_phone)} != {sum(adjusted_word2ph)}"  # fmt: skip

    # generated_phone から given_phone の間で音素が減った場合 (例: a, sh, i, t, a -> a, s, u) 、
    # adjusted_word2ph の要素の値が 1 未満になることがあるので、1 になるように値を増やす
    ## この時、adjusted_word2ph に記録されている音素数の合計を変えないために、
    ## 値を 1 にした分だけ右隣の要素から増やした分の差分を差し引く
    for adjusted_word2ph_element_index, adjusted_word2ph_element in enumerate(adjusted_word2ph):  # fmt: skip
        # もし現在の要素が 1 未満ならば
        if adjusted_word2ph_element < 1:
            # 値を 1 にするためにどれだけ足せばいいかを計算
            diff = 1 - adjusted_word2ph_element
            # adjusted_word2ph[(読み上げテキストの各文字のインデックス)] を 1 にする
            # これにより、当該文字に最低ラインとして 1 つの音素が割り当てられる
            adjusted_word2ph[adjusted_word2ph_element_index] = 1
            # 次の要素のうち、一番近くてかつ 1 以上の要素から diff を引く
            # この時、diff を引いた結果引いた要素が 1 未満になる場合は、その要素の次の要素の中から一番近くてかつ 1 以上の要素から引く
            # 上記を繰り返していって、diff が 0 になるまで続ける
            for i in range(1, len(adjusted_word2ph)):
                if adjusted_word2ph_element_index + i >= len(adjusted_word2ph):
                    break  # adjusted_word2ph の最後に達した場合は諦める
                if adjusted_word2ph[adjusted_word2ph_element_index + i] - diff >= 1:
                    adjusted_word2ph[adjusted_word2ph_element_index + i] -= diff
                    break
                else:
                    diff -= adjusted_word2ph[adjusted_word2ph_element_index + i] - 1
                    adjusted_word2ph[adjusted_word2ph_element_index + i] = 1
                    if diff == 0:
                        break

    # 逆に、generated_phone から given_phone の間で音素が増えた場合 (例: a, s, u -> a, sh, i, t, a) 、
    # 1文字あたり7音素以上も割り当てられてしまう場合があるので、最大6音素にした上で削った分の差分を次の要素に加える
    # 次の要素に差分を加えた結果7音素以上になってしまう場合は、その差分をさらに次の要素に加える
    for adjusted_word2ph_element_index, adjusted_word2ph_element in enumerate(adjusted_word2ph):  # fmt: skip
        if adjusted_word2ph_element > 6:
            diff = adjusted_word2ph_element - 6
            adjusted_word2ph[adjusted_word2ph_element_index] = 6
            for i in range(1, len(adjusted_word2ph)):
                if adjusted_word2ph_element_index + i >= len(adjusted_word2ph):
                    break  # adjusted_word2ph の最後に達した場合は諦める
                if adjusted_word2ph[adjusted_word2ph_element_index + i] + diff <= 6:
                    adjusted_word2ph[adjusted_word2ph_element_index + i] += diff
                    break
                else:
                    diff -= 6 - adjusted_word2ph[adjusted_word2ph_element_index + i]
                    adjusted_word2ph[adjusted_word2ph_element_index + i] = 6
                    if diff == 0:
                        break

    # この時点で given_phone の長さと adjusted_word2ph に記録されている音素数の合計が一致していない場合、
    # 正規化された読み上げテキストと given_phone が著しく乖離していることを示す
    # このとき、この関数の呼び出し元の get_text() にて InvalidPhoneError が送出される

    # 最初に削除した前後のダミー要素を追加して返す
    return [1] + adjusted_word2ph + [1]


def __g2phone_tone_wo_punct(text: str) -> list[tuple[str, int]]:
    """
    テキストに対して、音素とアクセント（0か1）のペアのリストを返す。
    ただし「!」「.」「?」等の非音素記号 (punctuation) は全て消える（ポーズ記号も残さない）。
    非音素記号を含める処理は `align_tones()` で行われる。
    また「っ」は「q」に、「ん」は「N」に変換される。
    例: "こんにちは、世界ー。。元気？！" →
    [('k', 0), ('o', 0), ('N', 1), ('n', 1), ('i', 1), ('ch', 1), ('i', 1), ('w', 1), ('a', 1), ('s', 1), ('e', 1), ('k', 0), ('a', 0), ('i', 0), ('i', 0), ('g', 1), ('e', 1), ('N', 0), ('k', 0), ('i', 0)]

    Args:
        text (str): テキスト

    Returns:
        list[tuple[str, int]]: 音素とアクセントのペアのリスト
    """

    prosodies = __pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)
    # logger.debug(f"prosodies: {prosodies}")
    result: list[tuple[str, int]] = []
    current_phrase: list[tuple[str, int]] = []
    current_tone = 0

    for i, letter in enumerate(prosodies):
        # 特殊記号の処理

        # 文頭記号、無視する
        if letter == "^":
            assert i == 0, "Unexpected ^"
        # アクセント句の終わりに来る記号
        elif letter in ("$", "?", "_", "#"):
            # 保持しているフレーズを、アクセント数値を 0-1 に修正し結果に追加
            result.extend(__fix_phone_tone(current_phrase))
            # 末尾に来る終了記号、無視（文中の疑問文は `_` になる）
            if letter in ("$", "?"):
                assert i == len(prosodies) - 1, f"Unexpected {letter}"
            # あとは "_"（ポーズ）と "#"（アクセント句の境界）のみ
            # これらは残さず、次のアクセント句に備える。
            current_phrase = []
            # 0 を基準点にしてそこから上昇・下降する（負の場合は上の `fix_phone_tone` で直る）
            current_tone = 0
        # アクセント上昇記号
        elif letter == "[":
            current_tone = current_tone + 1
        # アクセント下降記号
        elif letter == "]":
            current_tone = current_tone - 1
        # それ以外は通常の音素
        else:
            if letter == "cl":  # 「っ」の処理
                letter = "q"
            # elif letter == "N":  # 「ん」の処理
            #     letter = "n"
            current_phrase.append((letter, current_tone))

    return result


__PYOPENJTALK_G2P_PROSODY_A1_PATTERN = re.compile(r"/A:([0-9\-]+)\+")
__PYOPENJTALK_G2P_PROSODY_A2_PATTERN = re.compile(r"\+(\d+)\+")
__PYOPENJTALK_G2P_PROSODY_A3_PATTERN = re.compile(r"\+(\d+)/")
__PYOPENJTALK_G2P_PROSODY_E3_PATTERN = re.compile(r"!(\d+)_")
__PYOPENJTALK_G2P_PROSODY_F1_PATTERN = re.compile(r"/F:(\d+)_")
__PYOPENJTALK_G2P_PROSODY_P3_PATTERN = re.compile(r"\-(.*?)\+")


def __pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> list[str]:
    """
    ESPnet の実装から引用、概ね変更点無し。「ん」は「N」なことに注意。
    ref: https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    ------------------------------------------------------------------------------------------

    Extract phoneme + prosody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104
    """

    def _numeric_feature_by_regex(pattern: re.Pattern[str], s: str) -> int:
        match = pattern.search(s)
        if match is None:
            return -50
        return int(match.group(1))

    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = __PYOPENJTALK_G2P_PROSODY_P3_PATTERN.search(lab_curr).group(1)  # type: ignore
        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_E3_PATTERN, lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A1_PATTERN, lab_curr)
        a2 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A2_PATTERN, lab_curr)
        a3 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A3_PATTERN, lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_F1_PATTERN, lab_curr)

        a2_next = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A2_PATTERN, labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def __fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone_list` の tone（アクセントの値）を 0 か 1 の範囲に修正する。
    例: [(a, 0), (i, -1), (u, -1)] → [(a, 1), (i, 0), (u, 0)]

    Args:
        phone_tone_list (list[tuple[str, int]]): 音素とアクセントのペアのリスト

    Returns:
        list[tuple[str, int]]: 修正された音素とアクセントのペアのリスト
    """

    tone_values = set(tone for _, tone in phone_tone_list)
    if len(tone_values) == 1:
        assert tone_values == {0}, tone_values
        return phone_tone_list
    elif len(tone_values) == 2:
        if tone_values == {0, 1}:
            return phone_tone_list
        elif tone_values == {-1, 0}:
            return [(letter, 0 if tone == -1 else 1) for letter, tone in phone_tone_list]
        else:
            raise ValueError(f"Unexpected tone values: {tone_values}")
    else:
        raise ValueError(f"Unexpected tone values: {tone_values}")


def __handle_long(sep_phonemes: list[list[str]]) -> list[list[str]]:
    """
    フレーズごとに分かれた音素（長音記号がそのまま）のリストのリスト `sep_phonemes` を受け取り、
    その長音記号を処理して、音素のリストのリストを返す。
    基本的には直前の音素を伸ばすが、直前の音素が母音でない場合もしくは冒頭の場合は、
    おそらく長音記号とダッシュを勘違いしていると思われるので、ダッシュに対応する音素 `-` に変換する。

    Args:
        sep_phonemes (list[list[str]]): フレーズごとに分かれた音素のリストのリスト

    Returns:
        list[list[str]]: 長音記号を処理した音素のリストのリスト
    """

    for i in range(len(sep_phonemes)):
        if len(sep_phonemes[i]) == 0:
            # 空白文字等でリストが空の場合
            continue
        if sep_phonemes[i][0] == "ー":
            if i != 0:
                prev_phoneme = sep_phonemes[i - 1][-1]
                if prev_phoneme in VOWELS:
                    # 母音と「ん」のあとの伸ばし棒なので、その母音に変換
                    sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
                else:
                    # 「。ーー」等おそらく予期しない長音記号
                    # ダッシュの勘違いだと思われる
                    sep_phonemes[i][0] = "-"
            else:
                # 冒頭に長音記号が来ていおり、これはダッシュの勘違いと思われる
                sep_phonemes[i][0] = "-"
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]

    return sep_phonemes


__KATAKANA_PATTERN = re.compile(r"[\u30A0-\u30FF]+")
__MORA_PATTERN = re.compile("|".join(map(re.escape, sorted(MORA_KATA_TO_MORA_PHONEMES.keys(), key=len, reverse=True))))
__LONG_PATTERN = re.compile(r"(\w)(ー*)")


def __kata_to_phoneme_list(text: str) -> list[str]:
    """
    原則カタカナの `text` を受け取り、それをそのままいじらずに音素記号のリストに変換。
    注意点：
    - punctuation かその繰り返しが来た場合、punctuation たちをそのままリストにして返す。
    - 冒頭に続く「ー」はそのまま「ー」のままにする（`handle_long()` で処理される）
    - 文中の「ー」は前の音素記号の最後の音素記号に変換される。
    例：
    `ーーソーナノカーー` → ["ー", "ー", "s", "o", "o", "n", "a", "n", "o", "k", "a", "a", "a"]
    `?` → ["?"]
    `!?!?!?!?!` → ["!", "?", "!", "?", "!", "?", "!", "?", "!"]

    Args:
        text (str): カタカナのテキスト

    Returns:
        list[str]: 音素記号のリスト
    """

    if set(text).issubset(set(PUNCTUATIONS)):
        return list(text)
    # `text` がカタカナ（`ー`含む）のみからなるかどうかをチェック
    if __KATAKANA_PATTERN.fullmatch(text) is None:
        raise ValueError(f"Input must be katakana only: {text}")

    def mora2phonemes(mora: str) -> str:
        consonant, vowel = MORA_KATA_TO_MORA_PHONEMES[mora]
        if consonant is None:
            return f" {vowel}"
        return f" {consonant} {vowel}"

    spaced_phonemes = __MORA_PATTERN.sub(lambda m: mora2phonemes(m.group()), text)

    # 長音記号「ー」の処理
    long_replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))  # type: ignore
    spaced_phonemes = __LONG_PATTERN.sub(long_replacement, spaced_phonemes)

    return spaced_phonemes.strip().split(" ")


def __align_tones(phones_with_punct: list[str], phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    例: …私は、、そう思う。
    phones_with_punct:
        [".", ".", ".", "w", "a", "t", "a", "sh", "i", "w", "a", ",", ",", "s", "o", "o", "o", "m", "o", "u", "."]
    phone_tone_list:
        [("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), ("_", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0))]
    Return:
        [(".", 0), (".", 0), (".", 0), ("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), (",", 0), (",", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0), (".", 0)]

    Args:
        phones_with_punct (list[str]): punctuation を含む音素のリスト
        phone_tone_list (list[tuple[str, int]]): punctuation を含まない音素とアクセントのペアのリスト

    Returns:
        list[tuple[str, int]]: punctuation を含む音素とアクセントのペアのリスト
    """

    result: list[tuple[str, int]] = []
    tone_index = 0
    for phone in phones_with_punct:
        if tone_index >= len(phone_tone_list):
            # 余った punctuation がある場合 → (punctuation, 0) を追加
            result.append((phone, 0))
        elif phone == phone_tone_list[tone_index][0]:
            # phone_tone_list の現在の音素と一致する場合 → tone をそこから取得、(phone, tone) を追加
            result.append((phone, phone_tone_list[tone_index][1]))
            # 探す index を1つ進める
            tone_index += 1
        elif phone in PUNCTUATIONS:
            # phone が punctuation の場合 → (phone, 0) を追加
            result.append((phone, 0))
        else:
            logger.debug(f"phones: {phones_with_punct}")
            logger.debug(f"phone_tone_list: {phone_tone_list}")
            logger.debug(f"result: {result}")
            logger.debug(f"tone_index: {tone_index}")
            logger.debug(f"phone: {phone}")
            raise ValueError(f"Unexpected phone: {phone}")

    return result


def __distribute_phone(n_phone: int, n_word: int) -> list[int]:
    """
    左から右に 1 ずつ振り分け、次にまた左から右に1ずつ増やし、というふうに、
    音素の数 `n_phone` を単語の数 `n_word` に分配する。

    Args:
        n_phone (int): 音素の数
        n_word (int): 単語の数

    Returns:
        list[int]: 単語ごとの音素の数のリスト
    """

    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1

    return phones_per_word


class YomiError(Exception):
    """
    OpenJTalk で、読みが正しく取得できない箇所があるときに発生する例外。
    基本的に「学習の前処理のテキスト処理時」には発生させ、そうでない場合は、
    raise_yomi_error=False にしておいて、この例外を発生させないようにする。
    """
