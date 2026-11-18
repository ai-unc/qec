import unittest
from qec_discovery.transformer.stabilizer_tokenizer import encode, decode, pad_tokens
CODES = [["ZZI", "IZZ"], ["XXI", "IXX"], ["XXX", "ZZI"], ["ZZI", "IZZ", "XXX"], ["YIZ", "IYX"], ["ZII"],]

class StabilizerTokenizerTests(unittest.TestCase):
    def test_bad_letter(self):
        with self.assertRaises(ValueError):
            encode(["XIA"])
    def test_known_example(self):
        self.assertEqual(encode(["ZZI", "IZZ"]), [1, 7, 7, 4, 3, 4, 7, 7, 2])
    def test_all(self):
        for code in CODES:
            self.assertEqual(decode(encode(code)), code)
    def test_all_padding(self):
        for code in CODES:
            w_pad = pad_tokens(encode(code), 20)
            self.assertEqual(len(w_pad), 20)
            self.assertEqual(decode(w_pad), code)
    def test_mismatch_len(self):
        with self.assertRaises(ValueError):
            encode(["ZZI", "II"])
    def test_unknown_token_num(self):
        with self.assertRaises(ValueError):
            decode([1, 7, 8, 2])

if __name__ == "__main__":
    unittest.main()
