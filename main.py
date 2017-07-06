import sys
import simple_test
import motion_test

def main(argv):
    # simple_test.test_circ(1,2,10000, make_gif=False)
    motion_test.test_overfit_motion('overfit_test.txt', 'output/overfit_out.txt')

if __name__ == '__main__':
    main(sys.argv)
