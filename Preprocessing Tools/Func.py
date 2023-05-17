"""
Helper Function
"""
import pickle
import time
import datetime

class Func:
    # IO ------------------------------------------------
    @staticmethod
    def save_pkl(path,obj):
        with open(path, 'wb') as f:
            pickle.dump(obj,f)
    
    @staticmethod
    def load_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def df_str2list(data, attr):
        """
        @data DataFrame
        @attr list ['sentences','ent list']
        """
        print('Data Length : ', data.shape[0])

        for attr_name in attr:
            new_val = data[attr_name].tolist()
            new_val_list = []
            for i,line in enumerate(new_val):
                if type(line)!=str:
                    new_val_list.append([])
                else:
                    new_val_list.append(eval(line))
            data[attr_name] = new_val_list

    # data processing ------------------------------------------------
    @staticmethod
    def str2date(date_str):
        """
        %Y-%m-%d -> datetime.date
        """
        return datetime.datetime.strptime(date_str,'%Y-%m-%d')

    @staticmethod
    def date2str(date):
        """
        datetime.date -> %Y-%m-%d
        """
        return date.strftime('%Y-%m-%d')
    
    @staticmethod
    def date_scale(date1, date2):
        """
        @refer: https://blog.csdn.net/z564359805/article/details/80885801
        """
        date1=time.strptime(date1,"%Y-%m-%d")
        date2=time.strptime(date2,"%Y-%m-%d")

        date1=datetime.datetime(date1[0],date1[1],date1[2])
        date2=datetime.datetime(date2[0],date2[1],date2[2])

        return abs((date2-date1).days)+1
    
    @staticmethod
    def date_diff(date1, date2):
        """
        @refer: https://blog.csdn.net/z564359805/article/details/80885801
        """
        date1=time.strptime(date1,"%Y-%m-%d")
        date2=time.strptime(date2,"%Y-%m-%d")

        date1=datetime.datetime(date1[0],date1[1],date1[2])
        date2=datetime.datetime(date2[0],date2[1],date2[2])

        return (date2-date1).days