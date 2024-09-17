from typing import Any, Optional, Sequence

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature_onnx,
)


def __intersperse(lst: list[Any], item: Any) -> list[Any]:
    """
    リストの要素の間に特定のアイテムを挿入する
    style_bert_vits2.models.commons.intersperse と同一実装
    style_bert_vits2.models.commons モジュールは PyTorch に依存しているため、ONNX 推論時は import できない

    Args:
        lst (list[Any]): 元のリスト
        item (Any): 挿入するアイテム

    Returns:
        list[Any]: 新しいリスト
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_text_onnx(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    onnx_providers: list[str],
    onnx_provider_options: Optional[Sequence[dict[str, Any]]],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[
    NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
]:
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
        text,
        language_str,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = __intersperse(phone, 0)
        tone = __intersperse(tone, 0)
        language = __intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = extract_bert_feature_onnx(
        norm_text,
        word2ph,
        language_str,
        onnx_providers,
        onnx_provider_options,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == Languages.ZH:
        bert = bert_ori
        ja_bert = np.zeros((1024, len(phone)))
        en_bert = np.zeros((1024, len(phone)))
    elif language_str == Languages.JP:
        bert = np.zeros((1024, len(phone)))
        ja_bert = bert_ori
        en_bert = np.zeros((1024, len(phone)))
    elif language_str == Languages.EN:
        bert = np.zeros((1024, len(phone)))
        ja_bert = np.zeros((1024, len(phone)))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = np.array(phone)
    tone = np.array(tone)
    language = np.array(language)
    return bert, ja_bert, en_bert, phone, tone, language


def infer_onnx(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    language: Languages,
    hps: HyperParameters,
    onnx_session: onnxruntime.InferenceSession,
    onnx_providers: list[str],
    onnx_provider_options: Optional[Sequence[dict[str, Any]]],
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> NDArray[Any]:
    is_jp_extra = hps.version.endswith("JP-Extra")
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text_onnx(
        text,
        language,
        hps,
        onnx_providers=onnx_providers,
        onnx_provider_options=onnx_provider_options,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]

    x_tst = np.expand_dims(phones, axis=0)
    tones = np.expand_dims(tones, axis=0)
    lang_ids = np.expand_dims(lang_ids, axis=0)
    bert = np.expand_dims(bert, axis=0)
    ja_bert = np.expand_dims(ja_bert, axis=0)
    en_bert = np.expand_dims(en_bert, axis=0)
    x_tst_lengths = np.array([phones.shape[0]], dtype=np.int64)
    style_vec_tensor = np.expand_dims(style_vec, axis=0)
    del phones
    sid_tensor = np.array([sid], dtype=np.int64)

    input_names = [input.name for input in onnx_session.get_inputs()]
    output_name = onnx_session.get_outputs()[0].name
    if is_jp_extra:
        output = onnx_session.run(
            [output_name],
            {
                input_names[0]: x_tst,
                input_names[1]: x_tst_lengths,
                input_names[2]: sid_tensor,
                input_names[3]: tones,
                input_names[4]: lang_ids,
                input_names[5]: ja_bert,
                input_names[6]: style_vec_tensor,
                input_names[7]: length_scale,
                input_names[8]: sdp_ratio,
            },
        )
    else:
        raise NotImplementedError("Not implemented yet")

    audio = output[0][0, 0]

    del (
        x_tst,
        tones,
        lang_ids,
        bert,
        x_tst_lengths,
        sid_tensor,
        ja_bert,
        en_bert,
        style_vec,
    )  # , emo

    return audio