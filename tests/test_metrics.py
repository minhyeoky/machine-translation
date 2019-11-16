import unittest

from src.model.metric import compute_bleu


class MyTestCase(unittest.TestCase):
    def test_bleu(self):
        # https://donghwa-kim.github.io/BLEU.html
        pred_sentence = '빛이 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다'
        true_sentence = '빛이 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 가능성이 훨씬 높았다'
        pred_sentence, true_sentence = pred_sentence.split(), true_sentence.split()

        output = compute_bleu(reference_corpus=[[true_sentence]], translation_corpus=[pred_sentence], max_order=4, smooth=False)
        bleu_score, precisions, bp, ratio, translation_ratio, reference_length = output

        self.assertEqual(bleu_score, 0.25400289715190977)
        self.assertEqual(precisions,
                         [10/14, 5/13, 2/12, 1/11])
        self.assertEqual(translation_ratio, 14)

        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
