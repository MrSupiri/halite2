import random
import subprocess
import json
from datetime import datetime
import time
import os


def run(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    return output, p_status


def play(hm_bots=2):
    cmd = 'halite.exe -d "{}"'.format(random.choice(map_sizes))
    for ii in range(hm_bots):
        cmd += ' "{}"'.format(bots[ii])
    cmd += ' --replaydirectory "replays" -t -q'
    output, _ = run(cmd)

    return json.loads(output, encoding='ascii')


map_sizes = ['312 208',
             '240 160',
             '336 224',
             '360 240',
             '288 192',
             '264 176',
             '288 192']
bots = ['D:/virtualenv/DataScience/Scripts/python MyBot.py TrainingBot_1',
        'D:/virtualenv/DataScience/Scripts/python MyBot.py TrainingBot_2',
        'D:/virtualenv/DataScience/Scripts/python SentdeBot.py',
        'D:/virtualenv/DataScience/Scripts/python Training_Dummy.py']
AIWins = 0
BotWins = 0
i = 1
while True:
    winner = None
    if os.path.exists('models/TrainingBot_1.h5'):
        os.remove('models/TrainingBot_1.h5')
    if os.path.exists('models/TrainingBot_2.h5'):
        os.remove('models/TrainingBot_2.h5')

    if len([True for i in os.listdir('./') if i.endswith('.log')]) > 0:
        run('del *.log')
    if random.randint(0, 2) == 0:
        print('\n', datetime.now(), 'Starting 4 Player DeathMatch')
        log = play(4)
    else:
        print('\n', datetime.now(), 'Starting 1v1 DeathMatch')
        log = play(2)

    if log['error_logs']:
        print('There Was Error While Playing voiding the game in')
        for i in range(5, 0, -1):
            print(i)
            time.sleep(1)
        run('del *.log')
        continue

    if int(log['stats']['0']['rank']) == 1:
        if len(log['stats']) == 4:
            AIWins += 1
        winner = 'TrainingBot_1'
    elif int(log['stats']['1']['rank']) == 1:
        if len(log['stats']) == 4:
            AIWins += 1
        winner = 'TrainingBot_2'
    else:
        BotWins += 1

    try:
        p1_pct = round(AIWins / (AIWins + BotWins) * 100.0, 2)
        p2_pct = round(BotWins / (AIWins + BotWins) * 100.0, 2)
    except ZeroDivisionError:
        p1_pct = 0
        p2_pct = 0

    print("Games {}, AI Wins: {} WR: {}, Dummy Wins: {} WR: {}".format(i, AIWins, p1_pct,
                                                                       BotWins, p2_pct))

    for bot in log['stats']:
        p = log['stats'][bot]
        print('Bot #{}, Rank {}, Avg Response Time {}, Damage Dealt {}, Last Alive {}, Ship Count {}'
              .format(bot, p['rank'], round(float(p['average_frame_response_time']), 2), p['damage_dealt'],
                      p['last_frame_alive'], p['total_ship_count']))

    if winner is not None:
        if os.path.exists('models/{}.h5'.format(winner)):
            if os.path.exists('models/ProjectFaker.h5'):
                os.remove('models/ProjectFaker.h5')
            os.rename('models/{}.h5'.format(winner), 'models/ProjectFaker.h5')
            print('Model was Updated to {} weights'.format(winner))
    else:
        print('A dummy won the game shame on you Supiri -_-')

    i += 1

