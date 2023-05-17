"""
Extraction Tool for Location
"""
import pycountry
import MongoDB
import pandas as pd

class LocationFilter:
    """
    This class takes a list of strings and finds out the strings which are place names
    (country, region, city etc) or convert them to their offical name through GeoNames gazetteer
    """
    
    def __init__(self, host, user, passwd, port):
        self.db_mongodb = MongoDB(host = host, user = user, passwd = passwd, port = port) # use GeoNames database
        self.conn_ap = self.db_mongodb.connect("xxx", "xxx") # connect to mongodb database
        self.conn_ap_alter = self.db_mongodb.connect("xxx", "xxx")
    
    def is_a_country(self, ss): # recognize country by pycountry
        """
        @refer: LocationTagger
        """
        try:
            pycountry.countries.get(name=ss).alpha_3 # query name
            return True
        except AttributeError:
            try:
                pycountry.countries.get(official_name=ss).alpha_3 # query offical name
                return True
            except AttributeError:
                return False

    def batch_filter(self, raw_ents):
        """
        a batch version of geoNames_filter
        @param: 
        raw_ents a [list] of raw location string
        @return:
        ent a [list] of possible locations string
        """
        ent = []
        
        for raw_ent in raw_ents:
            ent.extend(self.geoNames_filter(raw_ent))
        
        return ent
                
    def geoNames_filter(self, raw_ent):
        """
        query raw_ent in GeoNames
        @param: 
        raw_ent [str] raw location
        @return:
        a [list] of possible locations
        """
        ent = raw_ent
        
        if len(raw_ent) > 3: # remain possible abbreviate
            # Only capitalize the first character
            # e.g. NEW YORK CITY to New York City
            ent = ' '.join([(i[0].upper()+i[1:].lower()) for i in raw_ent.split()])
        
        if self.is_a_country(ent): # if country in pycountry
            return [ent]
        
        # if regions or cities in geonames
        ap_cursor = self.conn_ap.find({"name" : ent}) # in AP
        df_return = pd.DataFrame(list(ap_cursor))
        if df_return.shape[0] != 0:
            return [ent]
            
        alter_cursor = self.conn_ap_alter.find({"alternatename" : raw_ent}) # in AP_alter(using raw_ent)
        df_return = pd.DataFrame(list(alter_cursor))
        if df_return.shape[0] != 0:
            return list(set(df_return['name'].values.tolist()))
        else:
            return []