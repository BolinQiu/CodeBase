######################################################
#                                                    #
#                 Log Information                    #
#                                                    #
######################################################

#  Log Structure

partition = " " # Select from [' ', '\t]，日志的实体之间是以什么分隔的


ENTITY1 = 0# 不同成分的位置
ENTITY2 = 2
EVENT_TYPE = 1
TIMESTAMP1 = 3
TIMESTAMP2 = 5

# Log Dictionary

raw_dir = "raw_log_dir" # 原始的日志文件，原始日志的处理函数还没有加进来，这一项可以先不填

# Logs to be analyzed
processed_dir = "case5.txt" # 当选择数据从.log文件导入时(from_dot = False)
dot_dir = "test.dot" # 当选择数据从.dot文件导入时(from_dot = True)
