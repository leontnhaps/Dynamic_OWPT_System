import unittest
from calibration.laser_center import summarize

class LaserCenterTests(unittest.TestCase):
    def row(self, u, v, **changes):
        return dict(dict(sample=str(u), session='a', distance_m=5, width=1296,
                         height=972, pan=0, tilt=0, screen='', ir=1, u=u, v=v), **changes)

    def test_mean_and_sample_std(self):
        result=summarize([self.row(10,20),self.row(12,24),self.row(14,28)])[0]
        self.assertEqual((result['count'],result['u_mean'],result['v_mean']), (3,12,24))
        self.assertEqual((result['u_std'],result['v_std']), (2,4))

    def test_conditions_separate_and_single_std_unknown(self):
        rows=[self.row(10,20), self.row(10,20,distance_m=4),
              self.row(10,20,width=1920), self.row(10,20,pan=5)]
        results=summarize(rows)
        self.assertEqual(len(results),4)
        self.assertTrue(all(r['u_std'] is None for r in results))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):summarize([])
