import sys
import simple_test

def main(argv):
    simple_test.test_circ(1,2,10000, make_gif=False)

if __name__ == '__main__':
    main(sys.argv)
