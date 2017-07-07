import sys
import simple_test
import motion_test

def main(argv):
    # simple_test.test_circ(1,2,10000, make_gif=False)
    # motion_test.test_overfit_motion('data/tests/overfit', 'output/overfit_out.txt')
    motion_test.test_bimodal_motion('data/tests/run','data/tests/walk', 'output/overfit_out.txt')

if __name__ == '__main__':
    main(sys.argv)
