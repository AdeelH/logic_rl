import numpy as np
import cv2


def viz_gridworld_policy(env, policy):
    for i, a in enumerate(policy._theta.argmax(axis=-1)):
        if i in env._obstacles:
            print('#', end=' ')
        elif i in env._waterStates:
            print('~', end=' ')
        else:
            print(['U', 'D', 'L', 'R'][a], end=' ')
        if (i+1) % 5 == 0:
            print()
    print('-------------')


def viz_cartpole(env, policy, max_iters=10000, delay_ms=40):
    '''Visualize a cartpole policy in action.
    
    Arguments:
        env -- Cartpole environment object
        policy -- Policy. Must have a samplAction() method.
        max_iters {int} -- maximum iterations
        delay_ms {int} -- milliseconds per frame
    '''
    PIXELS_PER_UNIT_X = 64
    BOUNDARY_DIST = env._x_lim
    H, W = 300, PIXELS_PER_UNIT_X * (BOUNDARY_DIST * 2 + 4)
    MID = W // 2
    
    IM_CART_Y = int(.75 * H)
    IM_CART_LEN = 10
    IM_POLE_LEN = 100

    img = np.empty((H, W, 3))
    reward = 0.
    for i in range(max_iters):
        if env.isEnd:
            if env._t >= env._t_lim:
                cv2.putText(img, 'SUCCESS!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 1., 0))
            else:
                cv2.putText(img, 'FAILURE!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 1.))
            cv2.imshow('cartpole', img)
            cv2.waitKey(0)
            break

        # step
        env.step(policy.samplAction(env.state))
        reward += (env.gamma**i) * env.reward
        x, theta = env._x, env._theta

        # clear image
        img.fill(1)

        # draw boundaries
        cv2.line(img, (0, IM_CART_Y), (W, IM_CART_Y), (.33, .33, .33), thickness=1)
        cv2.line(img, (MID - BOUNDARY_DIST * PIXELS_PER_UNIT_X, 0), (MID - BOUNDARY_DIST * PIXELS_PER_UNIT_X, H - 1), (0, 0, 0), thickness=4)
        cv2.line(img, (MID + BOUNDARY_DIST * PIXELS_PER_UNIT_X, 0), (MID + BOUNDARY_DIST * PIXELS_PER_UNIT_X, H - 1), (0, 0, 0), thickness=4)

        # draw cart
        im_cart_x = int(MID + PIXELS_PER_UNIT_X * x)
        cv2.line(img, (im_cart_x-IM_CART_LEN, IM_CART_Y), (im_cart_x+IM_CART_LEN, IM_CART_Y), (0, 0, 0), thickness=3)

        # draw pole
        im_pole_bottom = (im_cart_x, IM_CART_Y-3)
        im_pole_top = (int(im_cart_x + np.sin(theta)*IM_POLE_LEN), int(im_pole_bottom[1] - np.cos(theta)*IM_POLE_LEN))
        cv2.line(img, im_pole_bottom, im_pole_top, (0, 0, 1), thickness=2)

        # draw text boxes
        theta_txt_pos = (MID-70, 40)
        x_txt_pos = (im_cart_x-50, IM_CART_Y+40)
        cv2.putText(img, 'theta = %.4f (pi/10)' % (theta/env._theta_lim), theta_txt_pos, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        cv2.putText(img, 'x = %.4f' % (x), x_txt_pos, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        cv2.putText(img, 't = %.2f' % (env._t), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        cv2.putText(img, 'r = %.2f' % (reward), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        cv2.imshow('cartpole', img)
        cv2.waitKey(delay_ms)

    cv2.destroyAllWindows()
