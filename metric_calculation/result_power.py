import torch
from metrics import All_Metrics
import json
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report

def test(mode, mae_thresh=None, mape_thresh=0.0):
    len_nums = 0

### y_pred_in --> y_pred_load   y_true_in --> y_true_load
### y_pred_out --> y_pred_pv    y_true_out --> y_true_pv
### add y_pred_wind

    y_pred_load = []
    y_true_load = []
    y_pred_pv = []
    y_true_pv = []
    y_pred_wind = []
    y_true_wind = []
    y_pred_net_load = []
    y_true_net_load = []

    y_true_load_regionlist = []
    y_pred_load_regionlist = []
    y_true_pv_regionlist = []
    y_pred_pv_regionlist = []
    y_pred_wind_regionlist = []
    y_true_wind_regionlist = []
    y_true_net_load_regionlist = []
    y_pred_net_load_regionlist = []

    index_all = 0

    # Retrieve all JSON files from a folder and sort them by filename
    file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".json")])

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        with open(file_path, "r") as file:
            data_t = json.load(file)

        for i in range(len(data_t)):
            i_data = data_t[i]
            y_load = np.array(i_data["y_load"])
            y_pv = np.array(i_data["y_pv"])
            y_wind = np.array(i_data["y_wind"])
            st_pre_load = np.array(i_data["st_pre_load"])
            st_pre_pv = np.array(i_data["st_pre_pv"])
            st_pre_wind = np.array(i_data["st_pre_wind"])
            y_net_load = y_load - y_pv
            st_pre_net_load = st_pre_load - st_pre_pv

            i4data_all = int(data_t[i]["id"].split('_')[6])
            if index_all != i4data_all :
                len_nums = len_nums + 1
                y_true_load_region = np.stack(y_true_load, axis=-1)
                y_pred_load_region = np.stack(y_pred_load, axis=-1)
                y_true_pv_region = np.stack(y_true_pv, axis=-1)
                y_pred_pv_region = np.stack(y_pred_pv, axis=-1)
                y_true_wind_region = np.stack(y_true_wind, axis=-1)
                y_pred_wind_region = np.stack(y_pred_wind, axis=-1)    
                y_true_net_load_region = np.stack(y_true_net_load, axis=-1)
                y_pred_net_load_region = np.stack(y_pred_net_load, axis=-1)

                y_true_load_regionlist.append(y_true_load_region)
                y_pred_load_regionlist.append(y_pred_load_region)
                y_true_pv_regionlist.append(y_true_pv_region)
                y_pred_pv_regionlist.append(y_pred_pv_region)
                y_true_wind_regionlist.append(y_true_wind_region)
                y_pred_wind_regionlist.append(y_pred_wind_region)
                y_true_net_load_regionlist.append(y_true_net_load_region)
                y_pred_net_load_regionlist.append(y_pred_net_load_region)

                y_pred_load = []
                y_true_load = []
                y_pred_pv = []
                y_true_pv = []
                y_pred_wind = []
                y_true_wind = []
                y_pred_net_load = []
                y_true_net_load = []
                index_all = i4data_all
            y_true_load.append(y_load)
            y_pred_load.append(st_pre_load)
            y_true_pv.append(y_pv)
            y_pred_pv.append(st_pre_pv)
            y_true_wind.append(y_wind)
            y_pred_wind.append(st_pre_wind)
            y_true_net_load.append(y_net_load)
            y_pred_net_load.append(st_pre_net_load)

            if (i == len(data_t) - 1 and idx == len(file_list) - 1):
                y_true_load_region = np.stack(y_true_load, axis=-1)
                print(y_true_load_region.shape)
                y_pred_load_region = np.stack(y_pred_load, axis=-1)
                y_true_pv_region = np.stack(y_true_pv, axis=-1)
                y_pred_pv_region = np.stack(y_pred_pv, axis=-1)
                y_true_wind_region = np.stack(y_true_wind, axis=-1)
                y_pred_wind_region = np.stack(y_pred_wind, axis=-1)
                y_true_net_load_region = np.stack(y_true_net_load, axis=-1)
                y_pred_net_load_region = np.stack(y_pred_net_load, axis=-1)
                y_true_load_regionlist.append(y_true_load_region)
                y_pred_load_regionlist.append(y_pred_load_region)
                y_true_pv_regionlist.append(y_true_pv_region)
                y_pred_pv_regionlist.append(y_pred_pv_region)
                y_true_wind_regionlist.append(y_true_wind_region)
                y_pred_wind_regionlist.append(y_pred_wind_region)
                y_true_net_load_regionlist.append(y_true_net_load_region)
                y_pred_net_load_regionlist.append(y_pred_net_load_region)
                y_pred_load = []
                y_true_load = []
                y_pred_pv = []
                y_true_pv = []
                y_pred_wind = []
                y_true_wind = []
                y_pred_net_load = []
                y_true_net_load = []
                
    print('len_nums', len_nums)

    y_true_load = np.stack(y_true_load_regionlist, axis=0)
    y_pred_load = np.stack(y_pred_load_regionlist, axis=0)
    y_true_pv = np.stack(y_true_pv_regionlist, axis=0)
    y_pred_pv = np.stack(y_pred_pv_regionlist, axis=0)
    y_true_wind = np.stack(y_true_wind_regionlist, axis=0)
    y_pred_wind = np.stack(y_pred_wind_regionlist, axis=0)
    y_true_net_load = np.stack(y_true_net_load_regionlist, axis=0)
    y_pred_net_load = np.stack(y_pred_net_load_regionlist, axis=0)
    y_pred_load, y_pred_pv, y_pred_wind = np.abs(y_pred_load), np.abs(y_pred_pv), np.abs(y_pred_wind)
    print(y_true_load.shape, y_pred_load.shape, y_true_pv.shape, y_pred_pv.shape, y_true_wind.shape, y_pred_wind.shape, y_true_net_load.shape, y_pred_net_load.shape)

    if mode == 'classification':
        test_classfication(y_true_load, y_pred_load, y_true_pv, y_pred_pv, y_true_wind, y_pred_wind,y_true_net_load, y_pred_net_load)
    else:
        for t in range(y_true_load.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_load[:, t, ...], y_true_load[:, t, ...], mae_thresh, mape_thresh, None)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_load, y_true_load, mae_thresh, mape_thresh, None)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))

        for t in range(y_true_load.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_pv[:, t, ...], y_true_pv[:, t, ...], mae_thresh, mape_thresh, None)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_pv, y_true_pv, mae_thresh, mape_thresh, None)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))
        
        for t in range(y_true_load.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_wind[:, t, ...], y_true_wind[:, t, ...], mae_thresh, mape_thresh, None)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_wind, y_true_wind, mae_thresh, mape_thresh, None)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))

        for t in range(y_true_load.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_net_load[:, t, ...], y_true_net_load[:, t, ...], mae_thresh, mape_thresh, None)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_net_load, y_true_net_load, mae_thresh, mape_thresh, None)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))


def test_classfication(y_true_load, y_pred_load, y_true_pv, y_pred_pv, y_true_wind, y_pred_wind,y_true_net_load, y_pred_net_load):

    for i in range(4):
        if i == 0:
            y_true = y_true_load
            y_pred = y_pred_load
        elif i == 1:
            y_true = y_true_pv
            y_pred = y_pred_pv
        elif i == 2:
            y_true = y_true_wind
            y_pred = y_pred_wind
        else:
            y_true = y_true_net_load
            y_pred = y_pred_net_load
        y_true[y_true > 1] = 1
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"MicroF1: {micro_f1:.2f}")
        print(f"MacroF1: {macro_f1:.2f}")
        print(f"f1 Score: {f1:.2f}")

################################ result path ################################

folder_path = '../result_test/STCILLM_7b_pv10_'

mode = 'regression'

test(mode)
