import unittest
import AudioFile
import Clusterer
import Segment
import numpy as np


class TestCluster(unittest.TestCase):

    def test_00(self):
        examples = []
        for x in xrange(16):
            audio = AudioFile.AudioFile(open('samples/test_00/'+str(x)+'.wav'))
            segment = Segment.Segment(audio.wave_form, audio.sr)
            segment.extract_features()
            examples.append(segment.get_features())
        labels = Clusterer.Clusterer.label(examples, 3)
        bin_count = np.bincount(labels)
        self.assertEqual(3, len(bin_count))
        self.assertTrue(np.array_equal(np.array([4, 4, 8]), np.sort(bin_count)))

    def test_00_2(self):
        examples = []
        for x in [4, 5, 7, 8, 9, 11, 12, 13, 15]:
            audio = AudioFile.AudioFile(open('samples/test_00/'+str(x)+'.wav'))
            segment = Segment.Segment(audio.wave_form, audio.sr)
            segment.extract_features()
            examples.append(segment.get_features())
        labels = Clusterer.Clusterer.label(examples, 2)
        bin_count = np.bincount(labels)
        self.assertEqual(2, len(bin_count))
        self.assertTrue(np.array_equal(np.array([3, 6]), np.sort(bin_count)))

    def test_01(self):
        examples = []
        for x in xrange(19):
            audio = AudioFile.AudioFile(open('samples/test_01/'+str(x)+'.wav'))
            segment = Segment.Segment(audio.wave_form, audio.sr)
            segment.extract_features()
            examples.append(segment.get_features())
        labels = Clusterer.Clusterer.label(examples, 4)
        bin_count = np.bincount(labels)
        self.assertEqual(4, len(bin_count))
        self.assertTrue(np.array_equal(np.array([3, 4, 4, 8]), np.sort(bin_count)))
