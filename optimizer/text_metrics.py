from __future__ import annotations

import re

_NO_SPACE_LANGUAGES = (
    r"[\u4e00-\u9fff"   # 中文
    r"\u3040-\u309f"     # 日文平假名
    r"\u30a0-\u30ff"     # 日文片假名
    r"\uac00-\ud7af"     # 韩文
    r"\u0e00-\u0eff"     # 泰文
    r"\u1000-\u109f"     # 缅甸文
    r"\u1780-\u17ff"     # 高棉文
    r"\u0900-\u0dff]"    # 天城文/南亚文字
)

_SPACE_SEPARATED_LANGUAGES = (
    r"^[a-zA-Z0-9\'"
    r"\u0400-\u04ff"     # 西里尔字母
    r"\u0370-\u03ff"     # 希腊字母
    r"\u0600-\u06ff"     # 阿拉伯字母
    r"\u0590-\u05ff"     # 希伯来字母
    r"\u0e00-\u0e7f"     # 泰文
    r"]+$"
)

def is_pure_punctuation(text: str) -> bool:
    """检查文本是否仅包含标点符号。

    Args:
        text: 待检测的文本

    Returns:
        True 表示文本仅含标点（无 word 字符），False 表示含有文字
    """
    return not re.search(r"\w", text, re.UNICODE)

def is_mainly_cjk(text: str, threshold: float = 0.5) -> bool:
    """判断是否主要为不使用空格的亚洲语言文本。

    包括：中日韩、泰文、缅甸文、高棉文、印地语等

    Args:
        text: 待检测的文本
        threshold: 阈值比例（默认 0.5，即超过 50%）

    Returns:
        True 表示主要为不使用空格的亚洲语言，False 表示其他
    """
    if not text:
        return False

    no_space_count = len(re.findall(_NO_SPACE_LANGUAGES, text))
    total_chars = len("".join(text.split()))

    return no_space_count / total_chars > threshold if total_chars > 0 else False

def is_space_separated_language(text: str) -> bool:
    """判断文本是否为需要空格分隔的语言。

    需要空格的语言包括：
    - 拉丁字母语言：英语、法语、德语、西班牙语等
    - 西里尔字母语言：俄语、乌克兰语、保加利亚语等
    - 希腊字母语言：希腊语
    - 阿拉伯字母语言：阿拉伯语、波斯语、乌尔都语等
    - 希伯来字母语言：希伯来语

    不需要空格的语言（返回 False）：
    - 中文、日文、韩文（CJK）
    - 泰文、缅甸文、高棉文等

    Args:
        text: 待检测的文本

    Returns:
        True 表示需要空格分隔，False 表示不需要
    """
    if not text:
        return False
    return bool(re.match(_SPACE_SEPARATED_LANGUAGES, text.strip()))

def count_words(text: str) -> int:
    """统计文本字符/单词数。

    按字符计数的语言（不使用空格分词）：
    - CJK (中文、日文、韩文)
    - 泰文、缅甸文、高棉文、印地语等

    按单词计数的语言（使用空格分词）：
    - 拉丁字母语言 (英语、法语、德语、西班牙语等)
    - 西里尔字母语言 (俄语、乌克兰语、保加利亚语等)
    - 希腊字母、阿拉伯字母、希伯来字母等

    混合文本处理：
    - 按字符计数的语言统计字符数
    - 按单词计数的语言统计单词数
    - 返回总和

    Args:
        text: 待统计的文本

    Returns:
        字符数 + 单词数
    """
    if not text:
        return 0

    # 统计不使用空格的语言的字符数（CJK + 泰文/缅甸文等）
    char_count = len(re.findall(_NO_SPACE_LANGUAGES, text))

    # 移除不使用空格的字符后，统计使用空格的语言的单词数
    word_text = re.sub(_NO_SPACE_LANGUAGES, " ", text)
    words = word_text.strip().split()
    word_count = len(words) if words and words != [""] else 0

    return char_count + word_count
