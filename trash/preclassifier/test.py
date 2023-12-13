import py_stringmatching as sm

# create an alphabetical tokenizer that returns a bag of tokens
alphabet_tok = sm.AlphabeticTokenizer()
# create an alphanumeric tokenizer
alnum_tok = sm.AlphanumericTokenizer()
# create a delimiter tokenizer using comma as a delimiter
delim_tok = sm.DelimiterTokenizer(delim_set=[" "])
# create a qgram tokenizer using q=3
qg3_tok = sm.QgramTokenizer(qval=3)
# create a whitespace tokenizer
ws_tok = sm.WhitespaceTokenizer()

s = "test string12 5823"
s2 = "hallo test"
# run all tokenizers on s
print(alphabet_tok.tokenize(s))
print(alnum_tok.tokenize(s))
print(delim_tok.tokenize(s))
print(qg3_tok.tokenize(s))
print(ws_tok.tokenize(s))


x = "string matching package"

y = "string matching library string"

# compute Jaccard score over sets of tokens of x and y, tokenized using whitespace
jac = sm.Jaccard()
print(jac.get_raw_score(ws_tok.tokenize(x), ws_tok.tokenize(y)))
