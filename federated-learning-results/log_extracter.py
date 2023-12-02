#after running all the clients we will have a directory with the structure of
# {local_log_path}/federated-learning-results/{BACKEND}/{strategy}/{N_CLIENT_INSTANCES}
#Backend is jax,tensorflow,torch
#strategy is FedAdaGard, FedAvg, FedAdam, FedAvgM
#N_CLIENT_INSTANCES is 2,10

#     each log file is like
# client_i_0b1b1a597730a5bcf_log.log
# client_i_00dd60b0952d6b1b9_log.log
# client_i_005c4e21e7c493564_log.log
# client_i_092e36790e6beffcc_log.log
# client_i_04834531a10f9a04c_log.log
# server_log.log

#Where the long string is the instance id

#example of a log file of a client
# connection (no certificates were passed)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:50:44,030 | connection.py:42 | ChannelConnectivity.IDLE
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:50:44,031 | connection.py:42 | ChannelConnectivity.CONNECTING
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:50:44,052 | connection.py:42 | ChannelConnectivity.TRANSIENT_FAILURE
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:50:44,254 | connection.py:139 | gRPC channel closed
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:50:54,263 | grpc.py:49 | Opened insecure gRPC connection (no certificates were passed)
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:50:54,263 | grpc.py:49 | Opened insecure gRPC connection (no certificates were passed)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:50:54,291 | connection.py:42 | ChannelConnectivity.IDLE
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:50:54,291 | connection.py:42 | ChannelConnectivity.IDLE
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:50:54,292 | connection.py:42 | ChannelConnectivity.READY
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:50:54,292 | connection.py:42 | ChannelConnectivity.READY
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:51:55,319 | client.py:82 | Instance i-0b1b1a597730a5bcf : CPU usage history during fit() : {'timestamp': datetime.datetime(2023, 11, 26, 13, 51, 55, 319153), 'cpu_utilization': 87.0}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:51:55,319 | client.py:82 | Instance i-0b1b1a597730a5bcf : CPU usage history during fit() : {'timestamp': datetime.datetime(2023, 11, 26, 13, 51, 55, 319153), 'cpu_utilization': 87.0}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:53:30,620 | client.py:104 | Instance i-0b1b1a597730a5bcf : CPU usage history during evaluate() : {'timestamp': datetime.datetime(2023, 11, 26, 13, 53, 30, 620630), 'cpu_utilization': 75.4}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:53:30,620 | client.py:104 | Instance i-0b1b1a597730a5bcf : CPU usage history during evaluate() : {'timestamp': datetime.datetime(2023, 11, 26, 13, 53, 30, 620630), 'cpu_utilization': 75.4}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:54:20,347 | client.py:82 | Instance i-0b1b1a597730a5bcf : CPU usage history during fit() : {'timestamp': datetime.datetime(2023, 11, 26, 13, 54, 20, 347037), 'cpu_utilization': 86.9}

#We need a log parser and then delete duplicated logs

#We need to get the following metrics from the log files
#1. CPU usage history during fit()
#2. CPU usage history during evaluate()
#3. The time it took to run the fit() function

#The log of server is like

# ax-FedAdaGrad-run | INFO flwr 2023-11-26 13:50:44,090 | app.py:162 | Starting Flower server, config: ServerConfig(num_rounds=10, round_timeout=None)
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:50:44,092 | app.py:175 | Flower ECE: gRPC server running (10 rounds), SSL is disabled
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:50:44,092 | server.py:89 | Initializing global parameters
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:50:44,092 | server.py:272 | Using initial parameters provided by strategy
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:50:44,092 | server.py:91 | Evaluating initial parameters
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:51:06,514 | server.py:94 | initial parameters (loss, other metrics): 2.3083078861236572, {'accuracy': 0.6989734172821045}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:51:06,514 | server.py:104 | FL starting
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:51:06,514 | server.py:222 | fit_round 1: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:52:07,958 | server.py:236 | fit_round 1 received 5 results and 0 failures
# jax-FedAdaGrad-run | WARNING flwr 2023-11-26 13:52:08,085 | fedavg.py:242 | No fit_metrics_aggregation_fn provided
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:52:49,345 | server.py:125 | fit progress: (1, 38922.67578125, {'accuracy': 0.6807291507720947}, 102.83068657400008)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:52:49,345 | server.py:173 | evaluate_round 1: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:53:31,696 | server.py:187 | evaluate_round 1 received 5 results and 0 failures
# jax-FedAdaGrad-run | WARNING flwr 2023-11-26 13:53:31,697 | fedavg.py:273 | No evaluate_metrics_aggregation_fn provided
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:53:31,697 | server.py:222 | fit_round 2: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:54:33,092 | server.py:236 | fit_round 2 received 5 results and 0 failures
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 13:55:14,491 | server.py:125 | fit progress: (2, 379447.53125, {'accuracy': 0.656166672706604}, 247.97691307900004)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:55:14,491 | server.py:173 | evaluate_round 2: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:55:56,902 | server.py:187 | evaluate_round 2 received 5 results and 0 failures
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 13:55:56,903 | server.py:222 | fit_round 3: strategy sampled 5 clients (out of 5)
# ...
#
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:09:05,455 | server.py:125 | fit progress: (8, 693945.8125, {'accuracy': 0.5420699119567871}, 1078.940573234)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:09:05,455 | server.py:173 | evaluate_round 8: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:09:47,868 | server.py:187 | evaluate_round 8 received 5 results and 0 failures
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:09:47,868 | server.py:222 | fit_round 9: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:11:11,374 | server.py:236 | fit_round 9 received 5 results and 0 failures
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:11:52,744 | server.py:125 | fit progress: (9, 694108.6875, {'accuracy': 0.5272135138511658}, 1246.2302874939996)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:11:52,745 | server.py:173 | evaluate_round 9: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:12:35,326 | server.py:187 | evaluate_round 9 received 5 results and 0 failures
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:12:35,326 | server.py:222 | fit_round 10: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:13:36,387 | server.py:236 | fit_round 10 received 5 results and 0 failures
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:13:58,521 | server.py:125 | fit progress: (10, 694142.5, {'accuracy': 0.5132575631141663}, 1372.0073066169998)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:13:58,522 | server.py:173 | evaluate_round 10: strategy sampled 5 clients (out of 5)
# jax-FedAdaGrad-run | DEBUG flwr 2023-11-26 14:14:40,445 | server.py:187 | evaluate_round 10 received 5 results and 0 failures
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,445 | server.py:153 | FL finished in 1413.9313074519996
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,446 | app.py:225 | app_fit: losses_distributed [(1, 11.935277938842773), (2, 15.066919326782227), (3, 15.066919326782227), (4, 15.066919326782227), (5, 15.066919326782227), (6, 15.066919326782227), (7, 15.066919326782227), (8, 15.066919326782227), (9, 15.066919326782227), (10, 15.066919326782227)]
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,461 | app.py:226 | app_fit: metrics_distributed_fit {}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,462 | app.py:227 | app_fit: metrics_distributed {}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,468 | app.py:228 | app_fit: losses_centralized [(0, 2.3083078861236572), (1, 38922.67578125), (2, 379447.53125), (3, 600308.375), (4, 669147.5), (5, 687167.0625), (6, 692347.0625), (7, 693579.5625), (8, 693945.8125), (9, 694108.6875), (10, 694142.5)]
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,476 | app.py:229 | app_fit: metrics_centralized {'accuracy': [(0, 0.6989734172821045), (1, 0.6807291507720947), (2, 0.656166672706604), (3, 0.6334936022758484), (4, 0.612500011920929), (5, 0.5930059552192688), (6, 0.5748563408851624), (7, 0.5579166412353516), (8, 0.5420699119567871), (9, 0.5272135138511658), (10, 0.5132575631141663)]}

#We need to get the following metrics from the log file of servers

#1. After each round the train evaluation loss and accuracy
#2. The time it took to run the fit() function on the server for each round for all the clients
#3. The time it took to run the evaluate() function on the server for each round for all the clients
#4. The losses we had on the server for each round for all the clients # jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,468 | app.py:228 | app_fit: losses_centralized [(0, 2.3083078861236572), (1, 38922.67578125), (2, 379447.53125), (3, 600308.375), (4, 669147.5), (5, 687167.0625), (6, 692347.0625), (7, 693579.5625), (8, 693945.8125), (9, 694108.6875), (10, 694142.5)]
#5. The accuracy we had on the server for each round for all the clients # jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,476 | app.py:229 | app_fit: metrics_centralized {'accuracy': [(0, 0.6989734172821045), (1, 0.6807291507720947), (2, 0.656166672706604), (3, 0.6334936022758484), (4, 0.612500011920929), (5, 0.5930059552192688), (6, 0.5748563408851624), (7, 0.5579166412353516), (8, 0.5420699119567871), (9, 0.5272135138511658), (10, 0.5132575631141663)]}

#We need to creates dataframes for each of the above metrics for server df and clients df
#the columns of the dataframes are the instance ids, the strategy, the number of clients, the backend, the round number. metric and metric value

#We need to create a dataframe for the server
#the server columns will be the strategy, the number of clients, the backend, the round number, the metric and the metric value


#after we have the dfs we need to create some visuals on how the rounds, the number of clients, the backend and the strategy affect the metrics

import os
import glob
import re
import math
import pandas as pd

local_log_path = "."
backends = ["jax", "tensorflow", "torch"]
strategies = [ "FedAvg", "FedAdam", "FedAvgM","FedAdaGrad"]
client_instances = ["2", "5","8"]

df = pd.DataFrame(columns=["strategy", "n_clients", "backend",'client_bool','client_num', "metric", "metric_value", "timestamp",'round'])


# Function to add a timestamp to the data list
#  Instance i-09a90e0be41d874a0 : CPU usage history during fit() : {'timestamp': datetime.datetime(2023, 11, 26, 13, 33, 46, 702970), 'cpu_utilization': 52.5}
#  Instance i-09a90e0be41d874a0 : CPU usage history during evaluate() : {'timestamp': datetime.datetime(2023, 11, 26, 13, 34, 25, 762959), 'cpu_utilization': 54.2}
import ast
from datetime import datetime

def add_timestamp(element,backend, strategy, instance,client_instances,server=False,i=0,fits_i=0):
    
    method = "evaluate"
    if "fit" in element:
        method = "fit"
    #lets get the timestamp , the cpu utilization
    #lett get the dict surrounded by {}
    dict_str = '{' + element.split("{")[1]

    split_dict = dict_str.split(": ")
    # print(split_dict)
    #lets get the timestamp
    timestamp_str = split_dict[1].split("(")[1].split(")")[0]
    #lets convert the timestamp
    timestamp = datetime.strptime(timestamp_str, '%Y, %m, %d, %H, %M, %S, %f')
    timestamp_formatted = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    #lets get the cpu utilization
    cpu_utilization_str = split_dict[2].split("}")[0]
    cpu_utilization = float(cpu_utilization_str)
    data_to_add = {"strategy": strategy, 
                   "n_clients": instance, 
                   "backend": backend,
                   'client_bool':server,
                   'client_num':i, 
                   'metric': f"cpu_utilization_{method}", 
                   "metric_value": cpu_utilization, 
                   "timestamp": timestamp_formatted,
                   'round':fits_i}
    
    # print(data_to_add)
    #lets add the data to the df
    df.loc[len(df)] = data_to_add

# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,446 | app.py:225 | app_fit: losses_distributed [(1, 11.935277938842773), (2, 15.066919326782227), (3, 15.066919326782227), (4, 15.066919326782227), (5, 15.066919326782227), (6, 15.066919326782227), (7, 15.066919326782227), (8, 15.066919326782227), (9, 15.066919326782227), (10, 15.066919326782227)]
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,461 | app.py:226 | app_fit: metrics_distributed_fit {}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,462 | app.py:227 | app_fit: metrics_distributed {}
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,468 | app.py:228 | app_fit: losses_centralized [(0, 2.3083078861236572), (1, 38922.67578125), (2, 379447.53125), (3, 600308.375), (4, 669147.5), (5, 687167.0625), (6, 692347.0625), (7, 693579.5625), (8, 693945.8125), (9, 694108.6875), (10, 694142.5)]
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,476 | app.py:229 | app_fit: metrics_centralized {'accuracy': [(0, 0.6989734172821045), (1, 0.6807291507720947), (2, 0.656166672706604), (3, 0.6334936022758484), (4, 0.612500011920929), (5, 0.5930059552192688), (6, 0.5748563408851624), (7, 0.5579166412353516), (8, 0.5420699119567871), (9, 0.5272135138511658), (10, 0.5132575631141663)]}
def add_fit_metrics(element,backend, strategy, instance,client_instances,server=False):
    if 'losses_distributed' in element:
        right_of_array = element.split("[")[1]
        # print(right_of_array)
        left_of_array = right_of_array.split("]")[0]
        # print(left_of_array)
        array_full = "["+left_of_array+"]"
        # print(array_full)
        array = ast.literal_eval(array_full)
        # print(array)
        # [(1, 11.935277938842773), (2, 15.066919326782227), (3, 15.066919326782227), (4, 15.066919326782227), (5, 15.066919326782227), (6, 15.066919326782227), (7, 15.066919326782227), (8, 15.066919326782227), (9, 15.066919326782227), (10, 15.066919326782227)]
        for i in range(len(array)):
            data_to_add = {"strategy": strategy, 
                   "n_clients": instance, 
                   "backend": backend,
                   'client_bool':False,
                   'client_num':None, 
                   'metric': f"losses_distributed",
                   "metric_value": array[i][1], 
                   "timestamp": None,
                   'round':array[i][0]}
            df.loc[len(df)] = data_to_add

    elif 'losses_centralized' in element:
        right_of_array = element.split("[")[1]
        # print(right_of_array)
        left_of_array = right_of_array.split("]")[0]
        # print(left_of_array)
        array_full = "["+left_of_array+"]"
        # print(array_full)
        array = ast.literal_eval(array_full)
        # print(array)
        # [(0, 2.3083078861236572), (1, 38922.67578125), (2, 379447.53125), (3, 600308.375), (4, 669147.5), (5, 687167.0625), (6, 692347.0625), (7, 693579.5625), (8, 693945.8125), (9, 694108.6875), (10, 694142.5)]
        for i in range(len(array)):
            data_to_add = {"strategy": strategy, 
                   "n_clients": instance, 
                   "backend": backend,
                   'client_bool':False,
                   'client_num':None, 
                   'metric': f"losses_centralized",
                   "metric_value": array[i][1], 
                   "timestamp": None,
                   'round':array[i][0]}
            df.loc[len(df)] = data_to_add
# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:14:40,476 | app.py:229 | app_fit: metrics_centralized {'accuracy': [(0, 0.6989734172821045), (1, 0.6807291507720947), (2, 0.656166672706604), (3, 0.6334936022758484), (4, 0.612500011920929), (5, 0.5930059552192688), (6, 0.5748563408851624), (7, 0.5579166412353516), (8, 0.5420699119567871), (9, 0.5272135138511658), (10, 0.5132575631141663)]}
    elif 'metrics_centralized' in element:
        right_of_dict = element.split("{")[1]
        left_of_dict = right_of_dict.split("}")[0]
        dict_full = "{"+left_of_dict+"}"
        dict = ast.literal_eval(dict_full)
        # print(dict)
        # {'accuracy': [(0, 0.6989734172821045), (1, 0.6807291507720947), (2, 0.656166672706604), (3, 0.6334936022758484), (4, 0.612500011920929), (5, 0.5930059552192688), (6, 0.5748563408851624), (7, 0.5579166412353516), (8, 0.5420699119567871), (9, 0.5272135138511658), (10, 0.5132575631141663)]}
        accuracy_array = dict['accuracy']
        for i in range(len(accuracy_array)):
            data_to_add = {"strategy": strategy, 
                   "n_clients": instance, 
                   "backend": backend,
                   'client_bool':False,
                   'client_num':None, 
                   'metric': f"metrics_centralized",
                   "metric_value": accuracy_array[i][1], 
                   "timestamp": None,
                   'round':accuracy_array[i][0]}
            df.loc[len(df)] = data_to_add

# jax-FedAdaGrad-run | INFO flwr 2023-11-26 14:11:52,744 | server.py:125 | fit progress: (9, 694108.6875, {'accuracy': 0.5272135138511658}, 1246.2302874939996)
def add_fit_progress(element,backend, strategy, instance,client_instances,server=False):
    after_parenthesis = element.split("(")[1]
    # print(after_parenthesis)
    before_parenthesis = after_parenthesis.split(")")[0]
    # print(before_parenthesis)
    split_by_comma = before_parenthesis.split(",")
    round_1 = split_by_comma[0]
    print(round_1)
    loss = split_by_comma[1]
    # print(loss)
    right_of_dict = split_by_comma[2]
    # print(right_of_dict)
    left_of_dict = right_of_dict.split("}")[0]
    # print(left_of_dict)
    accuracy = left_of_dict.split(": ")[1]
    # print(accuracy)
    time_after_starting = split_by_comma[3]
    # print(time_after_starting)
    data_to_add = {"strategy": strategy, 
                   "n_clients": instance, 
                   "backend": backend,
                   'client_bool':server,
                   'client_num':None, 
                   'metric': f"fit_progress_loss",
                   "metric_value": loss, 
                   "timestamp": time_after_starting,
                   'round':round_1}
    df.loc[len(df)] = data_to_add

    data_to_add = {"strategy": strategy,
                     "n_clients": instance,
                     "backend": backend,
                     'client_bool': server,
                     'client_num': None,
                     'metric': f"fit_progress_accuracy",
                     "metric_value": accuracy,
                     "timestamp": time_after_starting,
                     'round': round_1}
    df.loc[len(df)] = data_to_add


# Function to parse a single log file
def parse_log_file(file_path, backend, strategy, instance,client_instances,server=False,i=0):
    with open(file_path, 'r') as file:
        fits_i = 1
        lines_seen = set() # holds lines already seen
        for line in file:
            if line in lines_seen: continue
            lines_seen.add(line)
            # print(line)
            # Use regular expressions to extract information from each line
            # Example: match = re.search(your_pattern, line)
            # Process each match according to your needs
            # chekc if it contains 
            # "accuracy" , 
            # "app_fit: losses_centralized" ,
            #  "app_fit: metrics_centralized" ,
            #  "app_fit: metrics_distributed" , 
            # "app_fit: losses_distributed" ,
            # "timestamp"
            # "FL finished"
            words_we_want = ["FL finished","accuracy", "app_fit: losses_centralized", "app_fit: metrics_centralized", "app_fit: metrics_distributed", "app_fit: losses_distributed", "timestamp"]
            if any(word in line for word in words_we_want):
                #lets break the line from "|" and get the 4th element
                elements = line.split("|")
                #lets get the 4th element
                element = elements[3]
                # print(element)
                # I want the fit_i to be divided by and then rounded up
                fits_i_2 = math.ceil(fits_i/2) 
                if "{'timestamp'" in element:
                    add_timestamp(element,backend, strategy, instance,client_instances,server,i,fits_i_2)
                    fits_i+=1
                if "app_fit" in element:
                    add_fit_metrics(element,backend, strategy, instance,client_instances,server=server)
                if "fit progress" in element:
                    add_fit_progress(element,backend, strategy, instance,client_instances,server=server)

                
            
# print("Starting log extraction")

# Iterate over directories and log files
for backend in backends:
    for strategy in strategies:
        for instance in client_instances:
            directory = f"./federated-learning-results/{backend}/{strategy}/{instance}"
            # print(f"Processing directory {directory}")
            # #get all the log files in the directory 
            # files = glob.glob(f"{directory}/*.log")
            #for some reason this is not working lets try to get the files with os
            if os.path.isdir(directory):
                files = os.listdir(directory)
                # print(files)
                i=1
                for file in files:
                    if file.endswith(".log"):
                        if "server" in file:
                            # print(file)
                            local_log_path = os.path.join(directory, file)
                            # print(local_log_path)
                            parse_log_file(local_log_path, backend, strategy, instance,client_instances,server=True)
                        elif "client" in file:
                            # print(file)
                            local_log_path = os.path.join(directory, file)
                            # print(local_log_path)
                            parse_log_file(local_log_path, backend, strategy, instance,client_instances,server=False,i=i)
                            i+=1
print(df)

#sort by strategy, n_clients, backend, round
# df = df.sort_values(by=['strategy', 'n_clients', 'backend','round'])

df.to_csv("./federated-learning-results/eda_logs.csv", index=False)