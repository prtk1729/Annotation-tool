import json  
from glob import glob
import argparse
from datetime import date
today = date.today()



# =========CLI Parsing===============
arg_container = argparse.ArgumentParser(description='Specify the Operating System')

# should be optional arguments container.add_arguments
arg_container.add_argument('--is_os_win', '-Is_Operating_System_Windows', type=int, required=True,
                           help='Is your OS Windows? type "--is_os_win 1" else 0')

arg_container.add_argument('--initials', "-Annotator's_name_initials", type=str, required=True,
                           help='example: if your name is Prateek Pani, type "--initials pp"')

arg_container.add_argument('--run', '-Run', type=bool, required=True,
                           help='Do you want to run this file? (--run 1) else 0')

# container.parse_args() and store in args
args = arg_container.parse_args()
# ====================================


def reset_json_file():
    '''When r'''
    day, month, year = today.day, today.month, today.year
    # print(day)
    # print(day, month, year)
    # paths_of_images = glob('static/f_le/*.png')
    paths_of_images = glob('static/pool_Set/*.jpg')
    pool_idx_list = list(range(len(paths_of_images)))
    gn_list, pp_list, gv_list = [], [], []
    pp_dict, gn_dict, gv_dict = {}, {}, {}
    for digit in range(5):
        for i in range(3):
            if i == 0:
                pp_list.extend(pool_idx_list[ (i*200+(digit*600)) : ((i+1)*200+(digit*600)) ])
            elif i == 1:
                gv_list.extend(pool_idx_list[ (i*200+(digit*600)) : ((i+1)*200+(digit*600)) ])
            else:
                gn_list.extend(pool_idx_list[ (i*200+(digit*600)) : ((i+1)*200+(digit*600)) ])

    # pp_dict = {f"img_{int(le)}.jpg": 0 for le in pp_list}
    pp_dict = {f"img_{int(le)}.jpg": int(le)%5 for le in pp_list}
    gv_dict = {f"img_{int(le)}.jpg": 0 for le in gv_list}
    gn_dict = {f"img_{int(le)}.jpg": 0 for le in gn_list}

    time_log_dict = {"time_logs": []}

    # non-win OS
    if args.is_os_win == 0:

        if args.initials == 'pp':
            with open(f'your_file_{args.initials}.txt', 'w') as f:
                for item in pp_list:
                    f.write("%s\n" % item)

            with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/yest_inp_file.json", "w") as fp:
                json.dump(obj=pp_dict, fp=fp)

            file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
            file1.write('0')
            file1.close()

            with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/time_logs.json", "w") as fp:
                fp.write(json.dumps(time_log_dict))


        if args.initials == 'gv':
            with open(f'your_file_gv.txt', 'w') as f:
                for item in gv_list:
                    f.write("%s\n" % item)

            with open(f"StatsIO/gv/{day}_{month}_{year}/yest_inp_file.json", "w") as fp:
                json.dump(obj=gv_dict, fp=fp)

            file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
            file1.write('0')
            file1.close()

            with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/time_logs.json", "w") as fp:
                fp.write(json.dumps(time_log_dict))


        if args.initials == 'gn':
            with open(f'your_file_gn.txt', 'w') as f:
                for item in gn_list:
                    f.write("%s\n" % item)

            with open(f"StatsIO/gn/{day}_{month}_{year}/yest_inp_file.json", "w") as fp:
                json.dump(obj=gn_dict, fp=fp)

            file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
            file1.write('0')
            file1.close()

            with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/time_logs.json", "w") as fp:
                fp.write(json.dumps(time_log_dict))




# ------------------------
    # win OS
    else:
        if args.initials == 'pp':
            with open('your_file_pp.txt', 'w') as f:
                for item in pp_list:
                    f.write("%s\n" % item)

            with open(f"StatsI\\pp\\{day}_{month}_{year}\\yest_inp_file.json", "w") as fp:
                json.dump(obj=pp_dict, fp=fp)

            file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
            file1.write('0')
            file1.close()

            with open(f"StatsIO\\{args.initials}\\{day}_{month}_{year}\\time_logs.json", "w") as fp:
                fp.write(json.dumps(time_log_dict))


        if args.initials == 'gv':
            with open(f'your_file_gv.txt', 'w') as f:
                for item in gv_list:
                    f.write("%s\n" % item)

            with open(f"StatsIO\\gv\\{day}_{month}_{year}\\yest_inp_file.json", "w") as fp:
                json.dump(obj=gv_dict, fp=fp)

            file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
            file1.write('0')
            file1.close()

            with open(f"StatsIO\\{args.initials}\\{day}_{month}_{year}\\time_logs.json", "w") as fp:
                fp.write(json.dumps(time_log_dict))


        if args.initials == 'gn':
            with open(f'your_file_gn.txt', 'w') as f:
                for item in gn_list:
                    f.write("%s\n" % item)

            with open(f"StatsIO\\gn\\{day}_{month}_{year}\\yest_inp_file.json", "w") as fp:
                json.dump(obj=gn_dict, fp=fp)

            file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
            file1.write('0')
            file1.close()

            with open(f"StatsIO\\{args.initials}\\{day}_{month}_{year}\\time_logs.json", "w") as fp:
                fp.write(json.dumps(time_log_dict))

    print('\nSuccessfully reset the json file!! Can Start parsing again from scratch\n')
    return 

# uncomment when hard-reset
if args.run == 1:
    reset_json_file()



# json.loads