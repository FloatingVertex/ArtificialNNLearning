import unittest
import NNImplementation as nn
import problems
import genetic_algorithms

class TestNNImplementation(unittest.TestCase):
    def test_training_to_output_1(self):
        weights = nn.create_weights((2,),[(10,),(4,),(1,)])
        for _ in range(1000):
            weights = genetic_algorithms.train_weights(weights,problems.output_1,nn.run_with_relu)

            # eval new eights and print score
            eval_function = nn.run_with_weights(nn.run_with_relu, weights)
            score = (problems.output_1(eval_function))
            print(score)
        self.assertTrue(score > 9)

if __name__ == '__main__':
    unittest.main()