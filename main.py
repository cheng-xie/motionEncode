import sys
import simple_test
import motion_test

def main(argv):
    # simple_test.test_circ(1,2,10000, make_gif=False)
    # motion_test.test_overfit_motion('data/tests/overfit', 'output/overfit_out.txt')
    # motion_test.test_multimodal_motion(['data/tests/run','data/tests/walk'], 'output/overfit_out.txt')
    try:
        load_path = argv[1]
    except:
        load_path = None
    print(argv[0])
    # motion_test.test_multimodal_motion(['data/tests/run_turn','data/tests/walk_turn'], 'output/overfit_out.txt', argv[0], load_path)
    # motion_test.simple_test_multimodal_motion(['data/humanoid','data/tests/walk_turn', 'data/tests/run','data/tests/walk'], 'output/overfit_out.txt', argv[0], load_path)
    motion_test.simple_test_multimodal_motion(['data/humanoid/jog','data/humanoid/walk'], 'output/overfit_out.txt', argv[0], load_path)

if __name__ == '__main__':
    main(sys.argv[1:])
