#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of static use of Google Visualization Python API."""

__author__ = "Misha Seltzer"

import sys

class PageGen:
    page_template = """
    <html>
      <head>
      <title>%s</title>
        <script src="http://www.google.com/jsapi" type="text/javascript"></script>
        <script>
          google.load("visualization", "1", {packages:["table"]});
          google.load("visualization", "1", {packages:["corechart"]});

          google.setOnLoadCallback(drawView);
          function drawView() {
            %s
          }
        </script>
      </head>
      <body>
        %s
      </body>
    </html>
    """
    
    groupTmplt = """
        <H1 align="center">%s</H1>
        """
    textTmplt = """
        <p id=%s>
        %s
        </p>
        """
    compTmplt = """
        <H2>%s</H2>
        <div id="%s" align="left"></div>
        """
    jsTmplt = """
        var opt = %s
        var %s_vw = new google.visualization.%s(document.getElementById('%s'));
        %s_vw.draw(%s, opt)
        """
    jsViewTmplt = """
        var %s = new google.visualization.DataView(%s);
        %s
        var opt = %s
        var %s_vw = new google.visualization.%s(document.getElementById('%s'));
        %s_vw.draw(%s, opt)
        """
    jsDataTmplt = """
        var %s = new google.visualization.DataTable(%s);
        """
    
    divCode = []
    jsCode = []

    def __init__(self):
        self.divCode = []
        self.jsCode = []

    def addGroup(self, id, opt={}):
        opt1 = dict(opt)
        opt1.setdefault("title", "%s Header Goes Here" % id)
        opt1.setdefault("header", opt1["title"])
        self.divCode.append(self.groupTmplt % (opt1["header"]))

    def addDesc(self, id, desc, opt={}):
        opt1 = dict(opt)
        opt1.setdefault("title", "%s Header Goes Here" % id)
        opt1.setdefault("header", opt1["title"])
        self.divCode.append(self.textTmplt % (id, desc))

    def addData(self, id, data):
        js = self.jsDataTmplt % (id, data.ToJSon())
        self.jsCode.append(js)
        
    def addView(self, id, data, cols, viewType, opt={}):
        opt1 = dict(opt)
        opt1.setdefault("title", "%s Header Goes Here" % id)
        opt1.setdefault("header", opt1["title"])
        div = self.compTmplt % (opt1["header"], id)
        self.divCode.append(div)
        
        if (cols):
            colSet = ("%s.setColumns(%s)" % (id, list(cols))).replace("\"","")
        else:
            colSet = ""
        js = self.jsViewTmplt % (id, data, colSet, str(opt), id, viewType, id, id, id)    
        self.jsCode.append(js)
        
    def printPage(self, opt=None):
        clSep = "\n"
        print self.page_template % (opt["title"], clSep.join(self.jsCode), clSep.join(self.divCode))
    
def main():
    import gviz_api

    # Creating the data
    description = [("name", "string", "Name"),
                   ("salary", "number", "Salary"),
                   ("full_time", "boolean", "Full Time Employee")]
    data = [("Jim",   (800, "$800"),      False),
            ("Bob",   (7000, "$7,000"),   True ),
            ("Mike",  (10000, "$10,000"), True ),
            ("Alice", (12500, "$12,500"), True )]

    # Loading it into gviz_api.DataTable
    data_table = gviz_api.DataTable(description)
    data_table.LoadData(data)
    
    # Create Page
    pg = PageGen()
    
    pg.addData("gdata", data_table)
    
    pg.addGroup("gchart_example", {"title":"Google Chart Example"})
    pg.addView("gchart1", "gdata", [0,1], "LineChart", {"title":"LineChart Example","width":1200,"height":600})
    
    pg.addGroup("gtable_example", {"title":"Google Table Example"})
    pg.addView("gtable1", "gdata", None, "Table", {"title":"Table Example"})
    
    pg.printPage({"title":"GChart Example"})
    
if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('inFile', nargs=1, help="Choose the in file to use")
    # #parser.add_argument('outFile', nargs=1, help="Choose the out file to use")
    # args = parser.parse_args()
    # main( args.inFile[0] )
    main()
