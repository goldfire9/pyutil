import csv_reader
import gviz_api
import json

class CsvDataTable:
    sch = None
    jdata = None
    
    def __init__( self, infile, delimiter="," ):
        header, data = csv_reader.readWithHeader(infile, delimiter)
        #print "header\n", header
        
        sch = []
        for i in range(0,len(header)):
            hname = header[i][0]
            htype = header[i][1]
            if htype=="float" or htype=="int":
                tp = "number"
            elif htype=="bool":
                tp = "boolean"
            else:
                tp = "string"
            
            # if len(header[i])>2:
                # sch.append((hname, tp, header[i][2]))
            # else:
                # sch.append((hname, tp, hname))
            sch.append((hname,tp))
            
        #print sch
        #print data
        self.sch = sch
        self.jdata = json.loads(json.dumps(data))
    
    def toDataTable( self, cols=None ):
        if not cols:
            sch = self.sch
            data = self.jdata
        else:
            sch = [ self.sch[i] for i in cols ]
            data = [ [x[i] for i in cols] for x in self.jdata ]
            # print self.sch
            # print sch
            # print self.jdata
            # print data
        # Loading it into gviz_api.DataTable
        data_table = gviz_api.DataTable(sch)
        data_table.LoadData(data)
        return data_table
        