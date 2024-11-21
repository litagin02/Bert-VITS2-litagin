import os
from pathlib import Path

import unidic
from fugashi import build_dictionary  # type: ignore


def fugashi_user_dict(compiled_dict_path: str, tmp_csv_path: str):
    # unidic-pyのunidicのpath
    unidic_path_str = unidic.DICDIR
    unidic_path = Path(unidic_path_str)

    dicrc_path = unidic_path / "dicrc"
    dicrc_path_str = str(dicrc_path)

    # windows環境だとパスが途中でエスケープされるバグがある(windows環境で引数指定する人がいないのかも)
    if os.name == "nt":
        # コンパイル後の辞書のpath
        compiled_dict_path = compiled_dict_path.replace("\\", "\\\\")
        # csvファイルのパス
        tmp_csv_path = tmp_csv_path.replace("\\", "\\\\")

        unidic_path_str = unidic_path_str.replace("\\", "\\\\")

        dicrc_path_str = dicrc_path_str.replace("\\", "\\\\")

        # mecabは辞書のエンコード名が曖昧にでUTF8でもutf-8でも良いがシステム辞書とユーザー辞書で完全一致していないと動かない。unidic-pyのエンコードがutf8になっているのでビルド時の引数を変えてはいけない
        # なぜかリソースファイルが読み込まれず、先頭の引数はなぜかくっつけないと認識しない。
        build_dictionary(f"-r{dicrc_path_str} -d {unidic_path_str} -u {compiled_dict_path} -f utf8 -t utf8 {tmp_csv_path}")
    else:
        build_dictionary(f"-d {unidic_path_str} -u {compiled_dict_path} -f utf8 -t utf8 {tmp_csv_path}")

    # システム辞書が間違った情報でコンパイルされているのでユーザー辞書も間違った情報に合わせる
    # matrix.def に記載されている left_id.def と right_id.def のサイズが逆になっているため、
    # 実行時にファイルが壊れていると判断され、コスト推定の時点でエラーとなり、辞書ビルドに失敗する
    # matrix.defはmatrix.binのソースファイル？だが容量が4gbを超えるため読み書きや同梱ができない
    # コスト推定は飛ばしてコストはハードコードしたが、辞書の互換性チェック時left_id.defとright_id.defのサイズをチェックするのでそのままだとエラーが起きる
    # なので、ユーザー辞書のバイナリにunidic-pyのleft_id.defとright_id.defのサイズを直接書き込む
    # この現象について詳しく解説している記事を見つけたのでおいておく　https://zenn.dev/zagvym/articles/28056236903369
    compiled_dict = Path(compiled_dict_path).read_bytes()
    compiled_dict = compiled_dict[:16] + b"\x0a\x3d\00\x00\x1c\x3c\x00" + compiled_dict[23:]
    Path(compiled_dict_path).write_bytes(compiled_dict)
