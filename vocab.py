# 50-word early-reader vocabulary tier (Dolch pre-primer/primer style + concrete picture words)
VOCAB_50 = [
    "a","and","away","big","blue","can","come","down","find","for",
    "funny","go","help","here","in","is","it","jump","little","look",
    "make","me","my","not","one","play","red","run","said","see",
    "the","three","to","two","up","we","where","yellow","you",
    "cat","dog","bird","tree","house","bed","happy","stop","yes","no","on",
]
assert len(VOCAB_50) == 50, len(VOCAB_50)

# Words Google Speech Commands v0.02 covers directly (isolated 1s clips)
SC_WORDS_ALL = ["backward","bed","bird","cat","dog","down","eight","five","follow","forward",
    "four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right",
    "seven","sheila","six","stop","three","tree","two","visual","wow","yes","zero"]
SC_COVERED = [w for w in VOCAB_50 if w in SC_WORDS_ALL]
SC_OTHER = [w for w in SC_WORDS_ALL if w not in VOCAB_50]
