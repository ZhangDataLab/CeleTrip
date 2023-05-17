"""
Extraction tool for dates
"""
import re
import datetime
import datefinder
import time

import Func

class DateExtractor:
    """
    Based on NER Tools (SpaCy)
    """

    month_full_name = ['January','February','March','April','May','June','July','August','September','October','November','December']

    abbr_dict = {
        'Sun.':'Sunday', 'SUN.':'Sunday',
        'Mon.':'Monday', 'MON.':'Monday',
        'Tues.':'Tuesday', 'TUES.':'Tuesday',
        'Wed.':'Wednesday', 'WED.':'Wednesday',
        'Thur.':'Thursday', 'Thurs.':'Thursday',
        'THUR.':'Thursday', 'THURS.':'Thursday',
        'Fri.':'Friday', 'FRI.':'Friday',
        'Sat.':'Saturday', 'SAT.':'Saturday',
    }

    dic_day1 = ['day', 'days', 'month','year','tonight','morning','noon','afternoon','evening','today','yesterday','the day before yesterday','tomorrow','the day after tomorrow','night', 'a.m.', 'p.m.', 'pm', 'am']
    week_tour = ['startDay','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weeken_tour = ['week']
    other_dict_num = ['day', 'days', 'month', 'night', 'year', 'years']

    today_adv_dict = ['today', 'night', 'noon', 'tonight', 'a.m.', 'p.m.', 'pm', 'am']

    dic_num = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'last':-7,'next':7}  

    adv_time_list = dic_day1 + week_tour + weeken_tour

    modifier = ['last','next','after','before','later','earlier', 'ago', 'past']
    week_modifier = ['next','after','before','last']
    weeken_modifier = ['ago', 'next','after','before','last', 'past', 'later']

    seq_num = {'first':'1','second':'2','third':'3','forth':'4','fifth':'5','sixth':'6','seventh':'7','eighth':'8','ninth':'9','twelfth':'12','tenth':'10'}

    def __init__(self):
        """ init """

    def resolve_time(self, sentence, ent_list, pub_date, have_past_word):
        """
        @param
        sentence [str]
        ent_list For DATE / TIME
        pub_date Publish Date of News
        @return
        actual_day_list [list]
        day_list [list]
        """
        actual_day_list, day_list= [], []
        ent_text_list = [text for text, label in ent_list]
        
        if len(ent_text_list) == 0:
                return actual_day_list, day_list

        date_list = self.merge_date_list(ent_text_list, sentence)
        date_list = self.deabbrs(date_list)
        
        relative_date_list, other_date_list = self.split_date_list(sentence, date_list)

        actual_day_list, day_list = self.get_ambiguous_time(relative_date_list, pub_date, have_past_word) # need pub_date for relative expression
        concrete_actual_day_list, concrete_day_list = self.get_concrete_time(other_date_list, pub_date)
        
        actual_day_list = actual_day_list + concrete_actual_day_list
        day_list = day_list + concrete_day_list
        
        return actual_day_list, day_list

    def merge_date_list(self, ent_text_list, sentence):
        """
        @param
        ent_list[global]
        sen_sub
        @return
        date_list
        """
        date_list = []
        for ent in ent_text_list:
            date_list.append(ent.strip())
        
        # ['Sunday', '19:00pm'] -> ['Sunday 19:00pm']
        date_list_merge = ' '.join(date_list)
        if date_list_merge in sentence:
            date_list = [date_list_merge]

        # e.g. Events at (Tuesday) [DATE] or (Sunday) [DATE] (19:00pm) [TIME] or (Monday afternoon) [TIME]
        # ['Tuesday', 'Sunday 19:00pm', 'Monday afternoon']
        
        date_list_len = len(date_list)
        if date_list_len >= 2:
            cnt = 0
            num_new = 0
            new_date_list = []
            for i in range(len(date_list)):
                new_date_flag = date_list[cnt]
                for j in range(cnt+1,len(date_list)):
                    if (new_date_flag + ' ' + date_list[j]) in sentence:
                        new_date_flag = new_date_flag + ' ' + date_list[j]
                        cnt += 1
                    else:
                        break

                date_list[i] = new_date_flag
                cnt += 1
                num_new += 1
                if cnt == len(date_list) or i == (len(date_list)-1):
                    new_date_list = date_list[:num_new]
                    break
                  
            date_list = new_date_list[:]

        return date_list

    def deabbrs(self, date_list):
        """
        date_list[list[str]]
        @param
        date_list
        @return 
        date_list
        """

        for date_adv_i in range(len(date_list)):
            date_adv = date_list[date_adv_i]
            for abbr in DateExtractor.abbr_dict:
                if abbr in date_adv:
                    date_list[date_adv_i] = date_list[date_adv_i].replace(abbr, DateExtractor.abbr_dict[abbr])

            seq_num = re.findall('[0-9]{1,2}st|[0-9]{1,2}nd|[0-9]{1,2}rd|[0-9]{1,2}th', date_adv)
            for se in seq_num:
                date_list[date_adv_i] = date_list[date_adv_i].replace(se, se[:-2])
            self.replace_seq(date_list, date_adv_i)

        return date_list

    def replace_seq(self, date_list, date_adv_i):
        """
        @param
        date_list [list[str]]
        date_list_index [int] index of date_abv
        """
        for k in DateExtractor.seq_num.keys():
            if k in date_list[date_adv_i]:
                date_list[date_adv_i] = date_list[date_adv_i].replace(k, DateExtractor.seq_num[k])
    
    def split_date_list(self, sentence, date_list):
        """
        @return
        relative_date_list
        other_date_list
        """

        relative_date_set = set()
        exclude_date_set = set()
        date_list_split = [date_ent.lower().split() for date_ent in date_list]

        for adv_time in DateExtractor.adv_time_list:
            for date_ent_i in range(len(date_list)):
                if adv_time.lower() in date_list_split[date_ent_i]:
                    relative_date_set.add(date_list[date_ent_i])
                    if adv_time.lower() in DateExtractor.dic_day1:
                        exclude_date_set.add(date_list[date_ent_i])

        return list(relative_date_set), list(set(date_list) - exclude_date_set)

    def get_ambiguous_time(self, date_list, pub_date, have_past_word):
        """
        sen_sub
        pub_date
        @return
        actual_time_list, date_list, date_index_list
        """

        if date_list == []:
            return [], []

        actual_time_list = self.resolve_relative_time(date_list, pub_date, have_past_word)
        
        # filter out yyyy-mm-dd
        actual_day_list = []; day_list = []
        for i,(actual_time,date_flag) in enumerate(zip(actual_time_list, date_list)):
            if len(actual_time) == 10:
                actual_day_list.append(actual_time)
                day_list.append(date_flag)
        return actual_day_list, day_list

    def resolve_relative_time(self, date_list, pub_date, have_past_word):
        """
        @param
        date_list
        pub_date
        @return
        actual_time_list
        """
        pub_time = datetime.datetime.strptime(pub_date, "%Y-%m-%d")
        actual_time_list = []

        for adv_time_index in range(len(date_list)):

            actual_time = pub_date
            tmp_date = date_list[adv_time_index].lower()

            if 'yesterday' in tmp_date:
                if 'day before yesterday' in tmp_date:
                    actual_time = (pub_time + datetime.timedelta(days = -2)).strftime("%Y-%m-%d")
                else:
                    actual_time = (pub_time + datetime.timedelta(days = -1)).strftime("%Y-%m-%d")
            elif 'tomorrow' in tmp_date:
                if 'day after tomorrow' in tmp_date:
                    actual_time = (pub_time + datetime.timedelta(days = 2)).strftime("%Y-%m-%d")
                else:
                    actual_time = (pub_time + datetime.timedelta(days = 1)).strftime("%Y-%m-%d")
            
            actual_time_list.append(actual_time)
        
        week_index = int(pub_time.strftime("%w"))
        if week_index == 0:
            week_index = 7

        for index in range(len(actual_time_list)):

            actual_time = actual_time_list[index]
            tmp_date = date_list[index].lower()

            for j in range(1, len(DateExtractor.week_tour)):

                if DateExtractor.week_tour[j].lower() in tmp_date and not self.contain_week_modifier_adv(tmp_date) and not have_past_word:
                    actual_time = (pub_time + datetime.timedelta( days = j - week_index )).strftime("%Y-%m-%d")

                elif DateExtractor.week_tour[j].lower() in tmp_date and (have_past_word or self.contain_week_modifier_adv(tmp_date)):
                    if have_past_word and j <= week_index and 'last' not in tmp_date:
                        actual_time = (pub_time + datetime.timedelta(days = j - week_index)).strftime("%Y-%m-%d")
                    elif 'last' in tmp_date or 'before' in tmp_date or have_past_word:
                        actual_time = (pub_time + datetime.timedelta(days = (j - week_index + DateExtractor.dic_num['last']))).strftime("%Y-%m-%d")
                    elif 'next' in tmp_date or 'after' in tmp_date:
                        actual_time = (pub_time + datetime.timedelta(days = (j - week_index + DateExtractor.dic_num['next']))).strftime("%Y-%m-%d")
            actual_time_list[index] = actual_time
        
        dic_num_list = list(DateExtractor.dic_num.keys())[:-3]
        for index in range(len(actual_time_list)):

            actual_time = actual_time_list[index]
            tmp_date = date_list[index].lower()
            
            for j in range(len(DateExtractor.weeken_tour)):
                if DateExtractor.weeken_tour[j] in tmp_date and not self.contain_weeken_modifier_adv(tmp_date):
                    actual_time = pub_date # this week
                elif DateExtractor.weeken_tour[j] in tmp_date and ('this' in tmp_date):
                    actual_time = pub_date
                elif DateExtractor.weeken_tour[j] in tmp_date and ('ago' in tmp_date or 'last' in tmp_date or 'before' in tmp_date or 'past' in tmp_date):
                    value = 1 # default to 1
                    for num_k in dic_num_list:
                        if num_k in tmp_date:
                            value = DateExtractor.dic_num[num_k]
                    actual_time = (pub_time + datetime.timedelta(days = -value*7)).strftime("%Y-%m-%d")
                elif DateExtractor.weeken_tour[j] in tmp_date and ('next' in tmp_date or 'after' in tmp_date or 'later' in tmp_date):
                    value = 1
                    for num_k in dic_num_list:
                        if num_k in tmp_date:
                                value = DateExtractor.dic_num[num_k]
                    actual_time = (pub_time + datetime.timedelta(days = value*7)).strftime("%Y-%m-%d")
            actual_time_list[index] = actual_time

        for index in range(len(date_list)):
            tmp_date = date_list[index].lower()
            value = re.search(r"\b\d+\b", tmp_date)
            tmp_date_list = re.split(' |-', tmp_date)

            if len(set(DateExtractor.other_dict_num) & set(tmp_date_list)) == 0:
                continue

            re_flag = 0
            for week_flag in DateExtractor.week_tour:
                if week_flag.lower() in tmp_date or 'tomorrow' in tmp_date or 'yesterday' in tmp_date or 'this' in tmp_date:
                    re_flag = 1
                    break
            if re_flag == 1:
                continue

            if value == None:
                value = 1
                for num_k in dic_num_list:
                    if num_k in tmp_date:
                        value = DateExtractor.dic_num[num_k]
            else:
                value = int(value.group())

            for td in DateExtractor.other_dict_num:
                try:
                    if td in tmp_date_list and not self.contain_modifier_adv(tmp_date_list):
                        if len(tmp_date) > len(td):
                            actual_time_list[index] = pub_date
                        else:
                            actual_time_list[index] = ''
                    elif td in tmp_date_list and ('ago' in tmp_date_list or 'earlier' in tmp_date_list or 'before' in tmp_date_list or 'past' in tmp_date_list or 'last' in tmp_date_list):
                        td_i = tmp_date_list.index(td)
                        if 'earlier' in tmp_date_list and tmp_date_list.index('earlier') < td_i:
                            continue
                        if 'day' in tmp_date_list or 'days' in tmp_date_list:
                            actual_time_list[index] = (pub_time + datetime.timedelta(days =- value*1)).strftime("%Y-%m-%d")
                        elif 'month' in tmp_date_list:
                            actual_time_list[index] = (pub_time + datetime.timedelta(days =- value*30)).strftime("%Y-%m-%d")
                        elif 'year' in tmp_date_list or 'years' in tmp_date_list:
                            actual_time_list[index] = (pub_time + datetime.timedelta(days =- value*365)).strftime("%Y-%m-%d")
                        elif 'night' in tmp_date_list:
                            actual_time_list[index] = (pub_time + datetime.timedelta(days =- 1)).strftime("%Y-%m-%d")
                    elif td in tmp_date_list and ('later' in tmp_date_list or 'next' in tmp_date_list or 'after' in tmp_date_list):
                        td_i = tmp_date_list.index(td)
                        if 'later' in tmp_date_list and tmp_date_list.index('later') < td_i:
                            continue
                        if 'day' in tmp_date:
                            actual_time_list[index] = (pub_time + datetime.timedelta(days = value*1)).strftime("%Y-%m-%d")
                        elif 'night' in tmp_date_list:
                            actual_time_list[index] = (pub_time + datetime.timedelta(days = 1)).strftime("%Y-%m-%d")
                except Exception as e:
                    actual_time_list[index] = ''
            
        return actual_time_list

    def get_concrete_time(self, date_list, pub_date):
        """
        use datefinder to resolve absolute time
        date_list list[str]
        """
        concrete_actual_date_list, concrete_date_list = [], []
        
        for date_ent in date_list:
            today = datetime.datetime.now().strftime("%d")

            try:
                find_dates = datefinder.find_dates(date_ent)
                find_date = ''
                number_list = re.findall(r"\d+", date_ent)
                for date in find_dates:
                    find_date = datetime.datetime.strftime(date, "%Y-%m-%d")
                    break
                if find_date != '':
                    year = find_date[:4]
                    day = find_date[-2:]
                    if not year in date_ent:
                        find_date = self.replace_year_by_rules(find_date, pub_date)
                    if day == today:
                        if (not day in number_list) and (not str(int(day)) in number_list):
                            continue
                    concrete_actual_date_list.append(find_date)
                    concrete_date_list.append(date_ent)
            except Exception as e:
                pass

        return concrete_actual_date_list, concrete_date_list

    def replace_year_by_rules(self, find_date, pub_date):
        """
        :param find_date:
        :param pub_date:
        :return:
        """
        pub_date_year = pub_date[:4]
        find_date_cand = []; find_date_scale = []; min_scale_i = 0

        find_date_cand.append(str(int(pub_date_year)-1) + find_date[4:])
        find_date_cand.append(pub_date_year + find_date[4:])
        find_date_cand.append(str(int(pub_date_year)+1) + find_date[4:])

        for i in range(3):
            find_date_scale.append(Func.date_scale(find_date_cand[i], pub_date))

        min_scale_i = find_date_scale.index(min(find_date_scale))

        return find_date_cand[min_scale_i]

    def contain_week_modifier_adv(self, adv):
        """
        week_modifier：last, next, after, before
        adv [str/list]
        """
        adv_list = adv if (type(adv) == list) else adv.split()

        for m in DateExtractor.week_modifier:
            if m in adv_list:
                return True
        return False

    def contain_weeken_modifier_adv(self, adv):
        """
        adv [str/list]
        """
        adv_list = adv if (type(adv) == list) else adv.split()

        for m in DateExtractor.weeken_modifier:
            if m in adv_list:
                return True
        return False

    def contain_modifier_adv(self, adv):
        """
        adv [str/list]
        """
        adv_list = adv if (type(adv) == list) else adv.split()

        for m in DateExtractor.modifier:
            if m in adv_list:
                return True
        return False

















