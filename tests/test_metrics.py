import unittest

from src.model.metric import compute_bleu
from src.utils import get_bleu_score


class MyTestCase(unittest.TestCase):
    def test_bleu(self):
        # https://donghwa-kim.github.io/BLEU.html
        pred_sentence = '빛이 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다'
        true_sentence = '빛이 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 가능성이 훨씬 높았다'
        pred_sentence, true_sentence = pred_sentence.split(), true_sentence.split()

        output = compute_bleu(reference_corpus=[[true_sentence]], translation_corpus=[pred_sentence], max_order=4,
                              smooth=False)
        bleu_score, precisions, bp, ratio, translation_ratio, reference_length = output

        self.assertEqual(bleu_score, 0.25400289715190977)
        self.assertEqual(precisions,
                         [10 / 14, 5 / 13, 2 / 12, 1 / 11])
        self.assertEqual(translation_ratio, 14)

        # self.assertEqual(True, False)

    def test_utils_get_bleu(self):
        orig = ['<start> 안녕 나는 이민혁 . <end> .']
        sentence = ['안녕 나는 이민혁 ㅋ <end> z k']
        bleu_score = get_bleu_score(x=orig, y=sentence)
        self.assertNotEqual(bleu_score, 1.0)

        bleu_score = get_bleu_score(
            x=['<start> 나 는 <unk> 일 <unk> 고 <unk> 에 가요 . <end>', '<start> <unk> 는 <unk> 에 항상 <unk> <unk> 에 가요 . <end>'],
            y=['나 는 일어나 자마자 화장실 에 가요 . <end> <end> <end>', '선생 이 문장 이 이해 가 요 . <end> <end> <end>'])
        self.assertEqual(bleu_score, 0.2170999676365301)


if __name__ == '__main__':
    unittest.main()
