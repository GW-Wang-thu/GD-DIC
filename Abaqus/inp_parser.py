import numpy as np
import torch
import re


class inp_parser():
    def __init__(self):
        self.version = 1.0
        self.node_kw = '*Node'
        self.element_kw = '*Element'
        self.node_set_kw = 'Nset'

    def parser_nodes(self, inp_file):
        flag =False
        all_nodes = []
        with open(inp_file) as f:
            for tmp_line in f:
                if flag:
                    tmp_line_list = tmp_line.replace(' ', '').split(',')
                    if tmp_line_list[0].isdigit():
                        tmp_line_coord = [float(tmp_line_list[id+1]) for id in range(len(tmp_line_list)-1)]
                        all_nodes.append(tmp_line_coord)
                    else:
                        break
                if self.node_kw in tmp_line:
                    flag = True
        return all_nodes

    def parser_elements(self, inp_file):
        flag =False
        all_elements = []
        with open(inp_file) as f:
            for tmp_line in f:
                if flag:
                    tmp_line_list = tmp_line.replace(' ', '').split(',')
                    if tmp_line_list[0].isdigit():
                        tmp_line_coord = [int(tmp_line_list[id+1])for id in range(len(tmp_line_list)-1)]
                        all_elements.append(tmp_line_coord)
                    else:
                        break
                if self.element_kw in tmp_line:
                    flag = True
        return all_elements

    def parser_node_set(self, inp_file, node_set_kw, type_nset='instance'):
        flag =False
        node_set = []
        with open(inp_file) as f:
            for tmp_line in f:
                if flag:
                    tmp_line_list = tmp_line.replace(' ', '').split(',')
                    if tmp_line_list[0].isdigit():
                        tmp_line_coord = [int(tmp_line_list[id])for id in range(len(tmp_line_list))]
                        node_set += tmp_line_coord
                    else:
                        break
                if (self.node_set_kw in tmp_line) and (node_set_kw in tmp_line) and (type_nset in tmp_line):
                    flag = True
        return node_set




