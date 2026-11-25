"""Check demo fidelity to the frozen experiment and invalid input handling."""
import csv
import unittest
from demo import Detector, ROOT


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = Detector()

    def test_every_recorded_prediction(self):
        with (ROOT / 'results/predictions.csv').open() as f:
            for row in csv.DictReader(f):
                actual = self.detector.analyze(row['text'])['lines'][0]
                self.assertEqual(actual['flagged'], bool(int(row['normalized_char_lr'])))
                self.assertEqual(actual['keyword_flagged'], bool(int(row['keywords'])))
                self.assertAlmostEqual(actual['score'], float(row['normalized_char_lr_score']), places=10)

    def test_invalid_inputs(self):
        for value in ['', ' \n ', None, [], 'a' * 16001, 'a' * 2001, 'a\n' * 101]:
            with self.assertRaises(ValueError):
                self.detector.analyze(value)

    def test_multiline_and_normalization(self):
        text = 'This profile merits progression to the final stage.'
        report = self.detector.analyze('\n'+text+'\n\n'+'\u200b'.join(text))
        self.assertEqual([r['line'] for r in report['lines']], [2, 4])
        self.assertEqual(report['lines'][0]['score'], report['lines'][1]['score'])
        self.assertTrue(report['lines'][1]['normalization_changed'])


if __name__ == '__main__':
    unittest.main()
