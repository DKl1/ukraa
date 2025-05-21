"""
Defines default encoder selection for given language pairs.
This mapping is used to choose a suitable encoder model when none is specified.
"""

PREFERRED_ENCODER = {
    ("uk", "en"): "labse", ("en", "uk"): "labse",
    ("uk", "de"): "labse", ("de", "uk"): "labse",
    ("uk", "pl"): "labse", ("pl", "uk"): "labse",
    ("uk", "fr"): "labse", ("fr", "uk"): "labse",
    ("uk", "it"): "labse", ("it", "uk"): "labse",
    ("uk", "es"): "labse", ("es", "uk"): "labse",
    ("uk", "cs"): "labse", ("cs", "uk"): "labse",
    ("uk", "ro"): "labse", ("ro", "uk"): "labse",
    ("uk", "hu"): "labse", ("hu", "uk"): "labse",
    ("uk", "hr"): "labse", ("hr", "uk"): "labse",
    ("uk", "bg"): "labse", ("bg", "uk"): "labse",
    ("uk", "sr"): "labse", ("sr", "uk"): "labse",
    ("uk", "tr"): "labse", ("tr", "uk"): "labse",
    ("uk", "pt"): "labse", ("pt", "uk"): "labse",
    ("uk", "sk"): "labse", ("sk", "uk"): "labse",
    ("uk", "sl"): "labse", ("sl", "uk"): "labse",
    ("uk", "nl"): "labse", ("nl", "uk"): "labse",
    ("uk", "sv"): "labse", ("sv", "uk"): "labse",
    ("uk", "da"): "labse", ("da", "uk"): "labse",
    ("uk", "nb"): "labse", ("nb", "uk"): "labse",
    ("uk", "fi"): "labse", ("fi", "uk"): "labse",
    ("uk", "et"): "labse", ("et", "uk"): "labse",
    ("uk", "lt"): "labse", ("lt", "uk"): "labse",
    ("uk", "lv"): "labse", ("lv", "uk"): "labse",

    ("uk", "zh"): "laser", ("zh", "uk"): "laser",
    ("uk", "ja"): "laser", ("ja", "uk"): "laser",
    ("uk", "vi"): "laser", ("vi", "uk"): "laser",
    ("uk", "ar"): "laser", ("ar", "uk"): "laser",
    ("uk", "hi"): "laser", ("hi", "uk"): "laser",
    ("uk", "fa"): "laser", ("fa", "uk"): "laser",
    ("uk", "ka"): "laser", ("ka", "uk"): "laser",
    ("uk", "az"): "laser", ("az", "uk"): "laser",
    ("uk", "he"): "laser", ("he", "uk"): "laser",
    ("uk", "uz"): "laser", ("uz", "uk"): "laser",
    ("uk", "kk"): "laser", ("kk", "uk"): "laser",
    ("uk", "ms"): "laser", ("ms", "uk"): "laser",
    ("uk", "id"): "laser", ("id", "uk"): "laser",
    ("uk", "ta"): "laser", ("ta", "uk"): "laser",
    ("uk", "bn"): "laser", ("bn", "uk"): "laser",
}

DEFAULT_ENCODER = "labse"
