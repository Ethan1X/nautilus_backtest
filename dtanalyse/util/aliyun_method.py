from aligo import Aligo
import pandas as pd

if __name__ == '__main__':
    
    ali = Aligo()
    
    # user = ali.get_user()
    ll = ali.get_file_list()
    print(f'file list: {ll}')
    data_files = []
    # if ll:
    #     for file in ll:
    #         print(file.file_id, file.name, file.type, file.url, file.download_url)
    #         if "supplement_Feature_6mth_202312" in file.name:
    #             data_files.append(file)
    #             print(f'{file.name}, {file.file_id}, url:{file.url}, d_url:{file.download_url}, --\n, {file}, \n')
    #             # ali.download_file(file)
    #         if file.name == "data":
    #             dl = ali.get_file_list(file.file_id)
    #             #for data in dl:
    #             #    print(data.file_id, data.name, data.type)
    #             #    data_files.append(data)
    #     print('download:', data_files)
    #     ali.download_files(data_files)

    # ali.upload_folder("/data/jupyter/Datas/merged_feats/20240301_0415_s/eth_usdt_binance")

    # file = ali.get_folder_by_path("/data")
    # ali.download_folder(folder_file_id=file.file_id, local_folder="/data/jupyter/Datas")
    
    # file = ali.get_file_by_path("/supplement_Feature_6mth_202312.tar")
    # ali.download_file(file_id=file.file_id, local_folder="/data/jupyter/Datas")

