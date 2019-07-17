############################ Constant ##################################
NUMBER_CHAR = '0123456789'

INTAB_L = "áàảãạâấầẩẫậăắằẳẵặđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
INTAB_U = "ÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"

VNI_OUTTAB_L = [
    "a1", "a2", "a3", "a4", "a5",
    "a6", "a61", "a62", "a63", "a64", "a65",
    "a8", "a81", "a82", "a83", "a84", "a85",
    "d9",
    "e1", "e2", "e3", "e4", "e5",
    "e6", "e61", "e62", "e63", "e64", "e65",
    "i1", "i2", "i3", "i4", "i5",
    "o1", "o2", "o3", "o4", "o5",
    "o6", "o61", "o62", "o63", "o64", "o65",
    "o7", "o71", "o72", "o73", "o74", "o75",
    "u1", "u2", "u3", "u4", "u5",
    "u7", "u71", "u72", "u73", "u74", "u75",
    "y1", "y2", "y3", "y4", "y5",
]

VNI_OUTTAB_U = [
    "A1", "A2", "A3", "A4", "A5",
    "A6", "A61", "A62", "A63", "A64", "A65",
    "A8", "A81", "A82", "A83", "A84", "A85",
    "D9",
    "E1", "E2", "E3", "E4", "E5",
    "E6", "E61", "E62", "E63", "E64", "E65",
    "I1", "I2", "I3", "I4", "I5",
    "O1", "O2", "O3", "O4", "O5",
    "O6", "O61", "O62", "O63", "O64", "O65",
    "O7", "O71", "O72", "O73", "O74", "O75",
    "U1", "U2", "U3", "U4", "U5",
    "U7", "U71", "U72", "U73", "U74", "U75",
    "Y1", "Y2", "Y3", "Y4", "Y5",
]

INTAB = [w for w in INTAB_U + INTAB_L]
VNI_OUTTAB = VNI_OUTTAB_U + VNI_OUTTAB_L

UNICODE2VNI = {u:v for u,v in zip(INTAB, VNI_OUTTAB)}


def convert_word_to_vni(word, tone_end=False):
	word = [UNICODE2VNI[c] if c in UNICODE2VNI else c for c in word]
	word = ''.join(word)

	if tone_end:
		word_no_tone = ''
		tone_num = ''
		for c in word:
			if c in NUMBER_CHAR:
				tone_num += c
			else:
				word_no_tone += c

		tone_num = [n for n in tone_num]
		tone_num.sort()
		tone_num = ''.join(tone_num)

		word = word_no_tone + tone_num

	return word


def remove_tone(word):
	word = convert_word_to_vni(word, tone_end=False)
	word = [c for c in word if c not in NUMBER_CHAR]
	word = ''.join(word)
	return word


def split_word_tone(word):
	word = convert_word_to_vni(word, tone_end=False)

	word_no_tone = ''
	tone_num = ''
	for c in word:
		if c in NUMBER_CHAR:
			tone_num += c
		else:
			word_no_tone += c

	tone_num = [n for n in tone_num]
	tone_num.sort()
	tone_num = ''.join(tone_num)

	return word_no_tone, tone_num
