"""
Defines default encoder selection for given language pairs.
"""

PREFERRED_ENCODER = {
    # ----------------------
    #   LaBSE (Latin/Cyrillic)
    # ----------------------
    ("uk","af"): "labse", ("af","uk"): "labse",   # Afrikaans
    ("uk","am"): "labse", ("am","uk"): "labse",   # Amharic (Cyrillic‐compatible)
    ("uk","be"): "labse", ("be","uk"): "labse",   # Belarusian
    ("uk","bg"): "labse", ("bg","uk"): "labse",   # Bulgarian
    ("uk","bs"): "labse", ("bs","uk"): "labse",   # Bosnian
    ("uk","ca"): "labse", ("ca","uk"): "labse",   # Catalan
    ("uk","ceb"): "labse",("ceb","uk"): "labse",  # Cebuano
    ("uk","co"): "labse", ("co","uk"): "labse",   # Corsican
    ("uk","cs"): "labse", ("cs","uk"): "labse",   # Czech
    ("uk","cy"): "labse", ("cy","uk"): "labse",   # Welsh
    ("uk","da"): "labse", ("da","uk"): "labse",   # Danish
    ("uk","de"): "labse", ("de","uk"): "labse",   # German
    ("uk","el"): "labse", ("el","uk"): "labse",   # Greek
    ("uk","en"): "labse", ("en","uk"): "labse",   # English
    ("uk","eo"): "labse", ("eo","uk"): "labse",   # Esperanto
    ("uk","es"): "labse", ("es","uk"): "labse",   # Spanish
    ("uk","et"): "labse", ("et","uk"): "labse",   # Estonian
    ("uk","eu"): "labse", ("eu","uk"): "labse",   # Basque
    ("uk","fi"): "labse", ("fi","uk"): "labse",   # Finnish
    ("uk","fr"): "labse", ("fr","uk"): "labse",   # French
    ("uk","fy"): "labse", ("fy","uk"): "labse",   # Frisian
    ("uk","ga"): "labse", ("ga","uk"): "labse",   # Irish
    ("uk","gd"): "labse", ("gd","uk"): "labse",   # Scots Gaelic
    ("uk","gl"): "labse", ("gl","uk"): "labse",   # Galician
    ("uk","ha"): "labse", ("ha","uk"): "labse",   # Hausa
    ("uk","haw"): "labse",("haw","uk"): "labse",  # Hawaiian
    ("uk","hr"): "labse", ("hr","uk"): "labse",   # Croatian
    ("uk","hu"): "labse", ("hu","uk"): "labse",   # Hungarian
    ("uk","hy"): "labse", ("hy","uk"): "labse",   # Armenian
    ("uk","is"): "labse", ("is","uk"): "labse",   # Icelandic
    ("uk","it"): "labse", ("it","uk"): "labse",   # Italian
    ("uk","jv"): "labse", ("jv","uk"): "labse",   # Javanese
    ("uk","ku"): "labse", ("ku","uk"): "labse",   # Kurdish
    ("uk","la"): "labse", ("la","uk"): "labse",   # Latin
    ("uk","lb"): "labse", ("lb","uk"): "labse",   # Luxembourgish
    ("uk","lt"): "labse", ("lt","uk"): "labse",   # Lithuanian
    ("uk","lv"): "labse", ("lv","uk"): "labse",   # Latvian
    ("uk","mg"): "labse", ("mg","uk"): "labse",   # Malagasy
    ("uk","mi"): "labse", ("mi","uk"): "labse",   # Māori
    ("uk","mr"): "labse", ("mr","uk"): "labse",   # Marathi
    ("uk","mt"): "labse", ("mt","uk"): "labse",   # Maltese
    ("uk","ne"): "labse", ("ne","uk"): "labse",   # Nepali
    ("uk","nl"): "labse", ("nl","uk"): "labse",   # Dutch
    ("uk","no"): "labse", ("no","uk"): "labse",   # Norwegian (Bokmål/Nynorsk)
    ("uk","pl"): "labse", ("pl","uk"): "labse",   # Polish
    ("uk","pt"): "labse", ("pt","uk"): "labse",   # Portuguese
    ("uk","ro"): "labse", ("ro","uk"): "labse",   # Romanian
    ("uk","ru"): "labse", ("ru","uk"): "labse",   # Russian
    ("uk","sk"): "labse", ("sk","uk"): "labse",   # Slovak
    ("uk","sl"): "labse", ("sl","uk"): "labse",   # Slovenian
    ("uk","so"): "labse", ("so","uk"): "labse",   # Somali
    ("uk","sq"): "labse", ("sq","uk"): "labse",   # Albanian
    ("uk","sr"): "labse", ("sr","uk"): "labse",   # Serbian
    ("uk","st"): "labse", ("st","uk"): "labse",   # Sesotho
    ("uk","su"): "labse", ("su","uk"): "labse",   # Sundanese
    ("uk","sv"): "labse", ("sv","uk"): "labse",   # Swedish
    ("uk","sw"): "labse", ("sw","uk"): "labse",   # Swahili
    ("uk","tl"): "labse", ("tl","uk"): "labse",   # Tagalog
    ("uk","tr"): "labse", ("tr","uk"): "labse",   # Turkish
    ("uk","tk"): "labse", ("tk","uk"): "labse",   # Turkmen
    ("uk","uk"): "labse", ("uk","uk"): "labse",   # Ukrainian itself
    ("uk","wo"): "labse", ("wo","uk"): "labse",   # Wolof
    ("uk","xh"): "labse", ("xh","uk"): "labse",   # Xhosa
    ("uk","yi"): "labse", ("yi","uk"): "labse",   # Yiddish (Hebrew‐script)
    ("uk","yo"): "labse", ("yo","uk"): "labse",   # Yoruba
    ("uk","zu"): "labse", ("zu","uk"): "labse",   # Zulu
    # ----------------------
    #       LASER
    # ----------------------
    ("uk","ar"): "laser", ("ar","uk"): "laser",   # Arabic
    ("uk","bn"): "laser", ("bn","uk"): "laser",   # Bengali
    ("uk","bo"): "laser", ("bo","uk"): "laser",   # Tibetan
    ("uk","fa"): "laser", ("fa","uk"): "laser",   # Persian
    ("uk","he"): "laser", ("he","uk"): "laser",   # Hebrew
    ("uk","hi"): "laser", ("hi","uk"): "laser",   # Hindi
    ("uk","hmn"): "laser",("hmn","uk"): "laser",  # Hmong
    ("uk","id"): "laser", ("id","uk"): "laser",   # Indonesian
    ("uk","j a"): "laser", ("ja","uk"): "laser",  # Japanese
    ("uk","ka"): "laser", ("ka","uk"): "laser",   # Georgian
    ("uk","km"): "laser", ("km","uk"): "laser",   # Khmer
    ("uk","kn"): "laser", ("kn","uk"): "laser",   # Kannada
    ("uk","ko"): "laser", ("ko","uk"): "laser",   # Korean
    ("uk","lo"): "laser", ("lo","uk"): "laser",   # Lao
    ("uk","ms"): "laser", ("ms","uk"): "laser",   # Malay
    ("uk","my"): "laser", ("my","uk"): "laser",   # Burmese
    ("uk","pa"): "laser", ("pa","uk"): "laser",   # Punjabi
    ("uk","ta"): "laser", ("ta","uk"): "laser",   # Tamil
    ("uk","te"): "laser", ("te","uk"): "laser",   # Telugu
    ("uk","th"): "laser", ("th","uk"): "laser",   # Thai
    ("uk","ug"): "laser", ("ug","uk"): "laser",   # Uighur
    ("uk","uz"): "laser", ("uz","uk"): "laser",   # Uzbek
    ("uk","vi"): "laser", ("vi","uk"): "laser",   # Vietnamese
    ("uk","zh"): "laser", ("zh","uk"): "laser",   # Chinese
}

DEFAULT_ENCODER = "labse"
