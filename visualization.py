import time
import matplotlib.pyplot as plt

def show_progress(e_now, e_all):
    if e_now % (e_all // 10) == 0:
        percent = 100 * e_now // e_all
        print('{} episodes\t {}%'.format(e_now, percent))

def show_success(episode):
    print('Optimal policy generated after around {} episodes.'.format(episode))
