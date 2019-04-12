#from pyspark import SparkContext, SparkConf
#from pyspark.vislib.vispark import execGPU,execGPUseq,cacheGPU,recvGPU,shuffleGPU,actionGPU,execGPUseq2
#from pyspark.vislib.assist import profiler 

#import sys
#sys.path.append('/home/whchoi/Project/vispark/python/pyspark/vislib/')

 #from run_vispark import SparkContext
import numpy
from PIL import Image
import sys
import getpass
from hdfs import InsecureClient
import time
from socket import *

#import matplotlib.pyplot as plt

#from pyspark.vislib.gpu_worker import send_data, send_args, send_halo, recv_data, run_gpu, shuffle_halo, drop_cuda, send_count, recv_halo, gpu_persist, get_cache, clear_devicemem, send_data_seq
#from pyspark.vislib.worker_assist import generate_data_shape

#profile=True
msgsize = 4096 

def get_ib_name():
    host_name  = gethostname()
    return host_name.replace('emerald','ib')

def saveFile(data,path,length=-1):

    msg_tag = "Save %s %d "%(path,length)
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]
   
    #print msg_tag
    #print data_str

    return data + msg_tag


def send(data):

    #print data.sum()

    send_str = data.tostring() 
    data_len = len(send_str)

    #send_str += '0'*(msgsize - data_len%msgsize)
    lenn = len(send_str)/msgsize 
    lenn += 1

    host_name  = get_ib_name()
    clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
    clisock.connect((host_name,4949))

    msg_tag="SEND %d %d %s END "%(lenn,data_len,"uchar")
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]
   
    #print msg_tag
 
    clisock.send(msg_tag)
   
    clisock.send(send_str)
    clisock.shutdown(1)

    #print "send done"

    data_key = clisock.recv(msgsize)
    #print data_key

    clisock.close()

    return data_key

def send_new(data):
        
    if type(data) is str:
        send_str = data
    else : 
        send_str = data.tostring() 
    data_len = len(send_str)

    send_str += '0'*(msgsize - data_len%msgsize)
    lenn = len(send_str)/msgsize 

    host_name  = get_ib_name()
    #clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
    #clisock.connect((host_name,4949))

    msg_tag="SEND %d %d %s "%(lenn,data_len,"uchar")
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]
   
    #clisock.send(msg_tag)

    #clisock.send(send_str)
    #clisock.shutdown(1)

    #print "send done"

        
    #data_key = clisock.recv(msgsize)
    #print data_key

    #clisock.close()

    #return data_key
    
    #return [msg_tag,send_str]

    return msg_tag+send_str


def send_seq(data):
   
    send_str = ""
    for elem in data:
        #send_str += data.tostring() 
        send_str += elem
    data_len = len(send_str)

    send_str += '0'*(msgsize - data_len%msgsize)
    lenn = len(send_str)/msgsize 

    host_name  = get_ib_name()
    #clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
    #clisock.connect((host_name,4949))

    msg_tag="SEND %d %d %s "%(lenn,data_len,"uchar")
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]
   
    #clisock.send(msg_tag)

    #clisock.send(send_str)
    #clisock.shutdown(1)

    #print "send done"

    #data_key = clisock.recv(msgsize)
    #print data_key

    #clisock.close()

    #return data_key
    
    #return [msg_tag,send_str]

    return msg_tag+send_str


def send_seq2(data):
    
    send_str = ""
    for elem in data:
        #send_str += data.tostring() 
        send_str += elem
    data_len = len(send_str)

    #send_str += '0'*(msgsize - data_len%msgsize)
    lenn = len(send_str)/msgsize 

    host_name  = get_ib_name()
    #clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
    #clisock.connect((host_name,4949))

    msg_tag="SEQ %d %d %s "%(lenn,data_len,"uchar")
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]
   
    #clisock.send(msg_tag)

    #clisock.send(send_str)
    #clisock.shutdown(1)

    #print "send done"

    #data_key = clisock.recv(msgsize)
    #print data_key

    #clisock.close()

    #return data_key
    
    #return [msg_tag,send_str]

    return msg_tag+send_str


def fill2(data):
    
    send_str = ""
    for elem in data:
        #send_str += data.tostring() 
        send_str += elem

    return send_str 

def fill(data):
    
    send_str = ""
    for elem in data:
        #send_str += data.tostring() 
        send_str += elem
    data_len = len(send_str)

    #send_str += '0'*(msgsize - data_len%msgsize)
    lenn = len(send_str)/msgsize 

    host_name  = get_ib_name()
    
    clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
    clisock.connect((host_name,4949))
    
    head_tag = "StartF %d END "%(lenn+1)
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
  
    msg_tag="FILL %d %d %s "%(lenn,data_len,"uchar")
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]

    clisock.send(head_tag + msg_tag + send_str)

    #clisock.send(send_str)
    clisock.shutdown(1)
    clisock.close()
    #print "send done"
    
    
    while True:
        time.sleep(0.1)
 
        clisock = socket(AF_INET, SOCK_STREAM)
        clisock.connect((host_name,4949))
 
        msg_tag="CHECK %d %d %s "%(lenn,data_len,"uchar")
        msg_tag+='0'*msgsize
        msg_tag = msg_tag[:msgsize]

        clisock.send(head_tag + msg_tag + send_str)

        clisock.shutdown(1)

        ack= clisock.recv(msgsize)
        print ack
        if ack == "CHECKED":
            break 
        clisock.close()

    #data_key = clisock.recv(msgsize)
    #print data_key

    #clisock.close()

    #return data_key
    
    #return [msg_tag,send_str]

    return send_str

 
def func_dict_to_str(func_args):
    args_info =""
    data_str  =""

    for key in func_args.keys():
        elem = func_args[key]
        if type(elem) is int:
            elem_str = numpy.array(elem).astype(numpy.int32).tostring()
            elem_len = 1
            args_info += "%s "%key
            args_info += "int %d "%elem_len
            data_str  += elem_str           
 
        elif type(elem) is float:
            elem_str = numpy.array(elem).astype(numpy.float64).tostring()
            elem_len = 1
            args_info += "%s "%key
            args_info += "double %d "%elem_len
            data_str  += elem_str           

        elif type(elem) is str:
            elem_str = elem
            elem_len = len(str)
            args_info += "%s "%key
            args_info += "string %d "%elem_len
            data_str  += elem_str           

        elif type(elem) is numpy.ndarray:
            elem_flat = elem.ravel()
            elem_str = elem_flat.tostring()
            elem_len = elem_flat.size
            args_info += "%s "%key
            if type(elem_flat[0]) is numpy.int32:
                args_info +="int %d "%elem_len 
            elif type(elem_flat[0]) is numpy.int64:
                args_info +="int %d "%elem_len 
            elif type(elem_flat[0]) is numpy.float64:
                args_info +="double %d "%elem_len
            elif type(elem_flat[0]) is numpy.float32:
                args_info +="float %d "%elem_len 
            else:
                print "Warning ! %s is not supported "%type(elem) 
            data_str  += elem_str           

        elif type(elem) is list:
            elem_flat = numpy.array(elem).ravel()
            elem_str = elem_flat.tostring()
            elem_len = elem_flat.size
            args_info += "%s "%key
            if type(elem_flat[0]) is numpy.int32:
                args_info +="int %d "%elem_len 
            elif type(elem_flat[0]) is numpy.float64:
                args_info +="double %d "%elem_len 
            else:
                print "Warning ! %s is not supported "%type(elem) 
            data_str  += elem_str           

        else:
            print "Warning ! %s is not supported "%type(elem) 

    args_info += "args_end "
    #print args_info 

    return args_info , data_str

  

def func_args_to_str(func_args):
    args_info =""
    data_str  =""

    for elem in func_args:
        if type(elem) is int:
            elem_str = numpy.array(elem).astype(numpy.int32).tostring()
            elem_len = 1
            args_info += "int %d "%elem_len
            data_str  += elem_str           
 
        elif type(elem) is float:
            elem_str = numpy.array(elem).astype(numpy.float64).tostring()
            elem_len = 1
            args_info += "double %d "%elem_len
            data_str  += elem_str           

        elif type(elem) is str:
            elem_str = elem
            elem_len = len(str)
            args_info += "string %d "%elem_len
            data_str  += elem_str           

        elif type(elem) is numpy.ndarray:
            elem_flat = elem.ravel()
            elem_str = elem_flat.tostring()
            elem_len = elem_flat.size
            if type(elem_flat[0]) is numpy.int32:
                args_info +="int %d "%elem_len 
            elif type(elem_flat[0]) is numpy.float64:
                args_info +="double %d "%elem_len
            elif type(elem_flat[0]) is numpy.float32:
                args_info +="float %d "%elem_len 
            else:
                print "Warning ! %s is not supported "%type(elem) 
            data_str  += elem_str           

        elif type(elem) is list:
            elem_flat = numpy.array(elem).ravel()
            elem_str = elem_flat.tostring()
            elem_len = elem_flat.size
            if type(elem_flat[0]) is numpy.int32:
                args_info +="int %d "%elem_len 
            elif type(elem_flat[0]) is numpy.float64:
                args_info +="double %d "%elem_len 
            else:
                print "Warning ! %s is not supported "%type(elem) 
            data_str  += elem_str           

        else:
            print "Warning ! %s is not supported "%type(elem) 

    return args_info , data_str

def run(data,cuda_code,func_name,func_args,out_info,kernel_info,force_free = 0):
    #print gethostname(),func_args
    args_info, data_str = func_args_to_str(func_args)
      
    code_str = cuda_code
    code_len = len(code_str)
    code_str += '0'*(msgsize - code_len%msgsize)
    lenn1 = len(code_str)/msgsize
 
    data_len = len(data_str)
    data_str += '0'*(msgsize - data_len%msgsize)
    lenn2 = len(data_str)/msgsize 

    msg_tag="RUN %d %d %d %d %s %d %d %d "%(lenn1,code_len,lenn2,data_len,func_name,out_info,len(kernel_info),force_free)
    for elem in kernel_info:
        msg_tag+="%d "%(elem)
    msg_tag+=args_info
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]
   
    #print msg_tag
    #print data_str

    return data + msg_tag + code_str + data_str
    #return [args_info,data_str]

def recv_transfer_func(transfer_func,channel=0,host_name="192.168.1.11",port=5959):
    #print "Try recv transfer"
    
    clisock = socket(AF_INET, SOCK_STREAM)
    clisock.connect((host_name,port))

    
    #print "Connected"


    head_tag = "Start "
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
  
    #clisock.send(head_tag)
    #clisock.shutdown(1)

    import time

    tt1 = time.time()

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        #print "Recv",len(data)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    clisock.close()
    #print "clisock closed"
    data="".join(str_buf)
    tt2 = time.time()

    #print "Recv transfer_func", tt2-tt1

    data = numpy.fromstring(data,numpy.float32)

    #print len(data)
    curr = transfer_func[channel*(256*4):(channel+1)*(256*4)]
    #print curr.max(), curr.min()
    #print data.max(), data.min()

    #print curr
    #print data[:4]

    transfer_func[channel*(256*4):(channel+1)*(256*4)]=data

    return transfer_func 


def recv_mmtx_selection_list(sersock):
    #print "Try recv transfer"
    
    #clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.connect((host_name,port))

    clisock,addr = sersock.accept()
    
    #print "Connected" + str(addr[0]) + " : " +str(addr[1])

  
    #clisock.send(head_tag)
    #clisock.shutdown(1)

    import time

    tt1 = time.time()

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        #print "Recv",len(data)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    clisock.close()
    #print "clisock closed"
    data="".join(str_buf)
    tt2 = time.time()

    #print "Recv transfer_func", tt2-tt1

    data = numpy.fromstring(data,numpy.float32)

    #print len(data)
    #curr = transfer_func[channel*(256*4):(channel+1)*(256*4)]


    mmtx = data[:16].reshape((4,4))
    inv_mmtx = data[16:32].reshape((4,4))
    ray_direction = data[32:35]

    channel = int(data[35])
    print "update channel %d"%channel
    #print mmtx
    #print inv_mmtx
    #print ray_direction

    select_list = data[36:].astype(numpy.int32)
    print "Select_list", select_list.max(), select_list.min()
    print len(select_list)
 
    return mmtx,inv_mmtx,ray_direction,select_list


def recv_mmtx_info(sersock,transfer_func=None):
    #print "Try recv transfer"
    
    #clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.connect((host_name,port))

    clisock,addr = sersock.accept()
    
    #print "Connected" + str(addr[0]) + " : " +str(addr[1])

  
    #clisock.send(head_tag)
    #clisock.shutdown(1)

    import time

    tt1 = time.time()

    print "Recv start"

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        #print "Recv",len(data)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    clisock.close()
    #print "clisock closed"
    data="".join(str_buf)
    tt2 = time.time()
    
    print "Recv done"

    #print "Recv transfer_func", tt2-tt1

    mode = numpy.fromstring(data[:4],numpy.int32)

    print "Mode %d"%mode
    camera_info = numpy.fromstring(data[4:4+4*(16+16+3)],numpy.float32)

    mmtx = camera_info[:16].reshape((4,4))
    inv_mmtx = camera_info[16:32].reshape((4,4))
    ray_direction = camera_info[32:35]

    cur_read = 4+4*(16+16+3)
    data = data[cur_read:]

    print mmtx
    print inv_mmtx
    print ray_direction

    #Volume Rendering
    if mode == 0:
        data = numpy.fromstring(data,numpy.float32)
        channel = int(data[0])
 
        if channel >= 0:
            transfer_func[channel*(256*4):(channel+1)*(256*4)]=data[1:]

        else :
            transfer_func[:] = data[1:] 

        return mmtx,inv_mmtx,ray_direction,transfer_func

    #Cell Segmentation
    elif mode == 1:
        select_list = numpy.fromstring(data,numpy.float32).astype(numpy.int32)
        return mmtx,inv_mmtx,ray_direction,select_list
    # Shader Mode
    elif mode == 2:
        vert_len = int(numpy.fromstring(data[:4],numpy.int32))
        frag_len = int(numpy.fromstring(data[4:8],numpy.int32))

        #print "Given Length ", vert_len, frag_len

        #print data[8:]
        
        vert_str = data[8:8+vert_len]
        frag_str = data[8+vert_len:]
   
        #print len(vert_str),len(frag_str)
        
 
        return mmtx,inv_mmtx,ray_direction,vert_str,frag_str
    elif mode == 3:
        #print data
        max_iter = int(numpy.fromstring(data[:4],numpy.int32))
        num_label = int(numpy.fromstring(data[4:8],numpy.int32))

        label_value = numpy.fromstring(data[8:8+num_label*4],numpy.float32)
        color_value = numpy.fromstring(data[8+num_label*4:],numpy.float32)

        return mmtx,inv_mmtx,ray_direction,max_iter,num_label,label_value,color_value

    #   if transfer_func is not None:
    #
    #      if channel >= 0:
    #          transfer_func[channel*(256*4):(channel+1)*(256*4)]=data[36:]
    #      else :
    #          print len(data[36:])
    #          print min(data[36:])
    #          print max(data[36:])
    #          transfer_func[:]=data[36:]
    #
    #  else :
    #      if channel == 100:
    #          flag = 0
    #      transfer_func = data[36:].astype(numpy.int32)
    #

    #return mmtx,inv_mmtx,ray_direction,transfer_func,flag 





def recv_mmtx_transfer_func_multi(sersock,transfer_func=None):
    #print "Try recv transfer"
    
    #clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.connect((host_name,port))

    clisock,addr = sersock.accept()
    
    #print "Connected" + str(addr[0]) + " : " +str(addr[1])

  
    #clisock.send(head_tag)
    #clisock.shutdown(1)

    import time

    tt1 = time.time()

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        #print "Recv",len(data)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    clisock.close()
    #print "clisock closed"
    data="".join(str_buf)
    tt2 = time.time()

    #print "Recv transfer_func", tt2-tt1

    data = numpy.fromstring(data,numpy.float32)

    #print len(data)
    #curr = transfer_func[channel*(256*4):(channel+1)*(256*4)]
    flag = 1

    mmtx = data[:16].reshape((4,4))
    inv_mmtx = data[16:32].reshape((4,4))
    ray_direction = data[32:35]

    channel = int(data[35])
    print "update channel %d"%channel
    #print mmtx
    #print inv_mmtx
    #print ray_direction

    if transfer_func is not None:

        if channel >= 0:
            transfer_func[channel*(256*4):(channel+1)*(256*4)]=data[36:]
        else :
            print len(data[36:])
            print min(data[36:])
            print max(data[36:])
            transfer_func[:]=data[36:]

    else :
        if channel == 100:
            flag = 0
        transfer_func = data[36:].astype(numpy.int32)
        

    return mmtx,inv_mmtx,ray_direction,transfer_func,flag 




def recv_mmtx_transfer_func1(sersock,transfer_func,channel=0):
    #print "Try recv transfer"
    
    #clisock = socket(AF_INET, SOCK_STREAM)
    #clisock.connect((host_name,port))

    clisock,addr = sersock.accept()
    
    #print "Connected" + str(addr[0]) + " : " +str(addr[1])

  
    #clisock.send(head_tag)
    #clisock.shutdown(1)

    import time

    tt1 = time.time()

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        #print "Recv",len(data)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    clisock.close()
    #print "clisock closed"
    data="".join(str_buf)
    tt2 = time.time()

    #print "Recv transfer_func", tt2-tt1

    data = numpy.fromstring(data,numpy.float32)

    #print len(data)
    #curr = transfer_func[channel*(256*4):(channel+1)*(256*4)]


    mmtx = data[:16].reshape((4,4))
    inv_mmtx = data[16:32].reshape((4,4))
    ray_direction = data[32:35]

    #print mmtx
    #print inv_mmtx
    #print ray_direction
    transfer_func[channel*(256*4):(channel+1)*(256*4)]=data[35:]

    return mmtx,inv_mmtx,ray_direction,transfer_func 



def recv_mmtx_transfer_func(transfer_func,channel=0,host_name="192.168.1.11",port=5959):
    #print "Try recv transfer"
    
    clisock = socket(AF_INET, SOCK_STREAM)
    clisock.connect((host_name,port))

    
    #print "Connected"


    head_tag = "Start "
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
  
    #clisock.send(head_tag)
    #clisock.shutdown(1)

    import time

    tt1 = time.time()

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        #print "Recv",len(data)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    clisock.close()
    #print "clisock closed"
    data="".join(str_buf)
    tt2 = time.time()

    #print "Recv transfer_func", tt2-tt1

    data = numpy.fromstring(data,numpy.float32)

    #print len(data)
    curr = transfer_func[channel*(256*4):(channel+1)*(256*4)]


    mmtx = data[:16].reshape((4,4))
    inv_mmtx = data[16:32].reshape((4,4))
    ray_direction = data[32:35]

    #print mmtx
    #print inv_mmtx
    #print ray_direction
    transfer_func[channel*(256*4):(channel+1)*(256*4)]=data[35:]

    return mmtx,inv_mmtx,ray_direction,transfer_func 



def recv(data):


    host_name  = get_ib_name()
    clisock = socket(AF_INET, SOCK_STREAM)
    clisock.connect((host_name,4949))


    msg_tag ="RECV %s END "%("fake")
    msg_tag += '0'*msgsize
    msg_tag = msg_tag[:msgsize]
 
    lenn = (len(data) +len(msg_tag))/msgsize

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
  
    
 
    clisock.send(head_tag+data+msg_tag)
    clisock.shutdown(1)
    
    """
    msg = clisock.recv(msgsize)
    print msg

    lenn = int(msg[:msg.find('**')])
    msg = msg[msg.find('**')+2:]
    data_len = int(msg[:msg.find('**')])
    """

    #lenn = 385
    #data_len = 786432


   
    #print "recv"
    #clisock.shutdown(1)

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    #clisock.shutdown(1)
    clisock.close()

    data="".join(str_buf)

    #print len(data), data[:10]


    return data




#def recv_transfer_func(transfer_func,channel=0,host_name="192.168.1.11",port=5959):
def viewer(data,bsize,dst="192.168.1.11",port=5959):


    host_name  = get_ib_name()
    clisock = socket(AF_INET, SOCK_STREAM)
    clisock.connect((host_name,4949))


    msg_tag ="VIEWER %d %s %d END "%(bsize,dst,port)
    msg_tag += '0'*msgsize
    msg_tag = msg_tag[:msgsize]
 
    lenn = (len(data) +len(msg_tag))/msgsize

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
  
    
 
    clisock.send(head_tag+data+msg_tag)
    clisock.shutdown(1)
    
    """
    msg = clisock.recv(msgsize)
    print msg

    lenn = int(msg[:msg.find('**')])
    msg = msg[msg.find('**')+2:]
    data_len = int(msg[:msg.find('**')])
    """

    #lenn = 385
    #data_len = 786432


   
    #print "recv"
    #clisock.shutdown(1)

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    #clisock.shutdown(1)
    clisock.close()

    data="".join(str_buf)

    #print len(data), data[:10]


    return data

def getnodelist(node_file):
    node_list = []
    
    with open(node_file) as f:
        lines =f.readlines()

        for elem in lines:
            elem = elem.strip()

            if elem.find('#')==0:
                continue
            if elem.find('\n')!=-1:
                elem = elem[:-2]

            if len(elem) > 0:
                #elem = elem.replace('ib','emerald')

                node_list.append(elem)

    return node_list


def gpuinit(node_file):
   

    node_list = []
    
    with open(node_file) as f:
        lines =f.readlines()

        for elem in lines:
            elem = elem.strip()

            if elem.find('#')==0:
                continue
            if elem.find('\n')!=-1:
                elem = elem[:-2]

            if len(elem) > 0:
                #elem = elem.replace('ib','emerald')

                node_list.append(elem)


 
    msg_tag ="init"

    msg_tag +=" END "
    msg_tag += '0'*msgsize

    lenn = len(msg_tag)/msgsize
    msg_tag = msg_tag[:lenn*msgsize] 

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
 
    for node in node_list:

        host_name  = node
        clisock = socket(AF_INET, SOCK_STREAM)
        clisock.connect((host_name,4949))

        clisock.send(head_tag+msg_tag)
        clisock.shutdown(1)
        clisock.close()


def gpukill(node_file):
   

    node_list = []
    
    with open(node_file) as f:
        lines =f.readlines()

        for elem in lines:
            elem = elem.strip()

            if elem.find('#')==0:
                continue
            if elem.find('\n')!=-1:
                elem = elem[:-2]

            if len(elem) > 0:
                #elem = elem.replace('ib','emerald')

                node_list.append(elem)


 
    msg_tag ="kill"

    msg_tag +=" END "
    msg_tag += '0'*msgsize

    lenn = len(msg_tag)/msgsize
    msg_tag = msg_tag[:lenn*msgsize] 

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
 
    for node in node_list:

        host_name  = node
        clisock = socket(AF_INET, SOCK_STREAM)
        clisock.connect((host_name,4949))

        clisock.send(head_tag+msg_tag)
        clisock.shutdown(1)
        clisock.close()


def newShuffle(send_info,node_list=None):
  
    if node_list == None: 
        node_list = ['ib%02d'%elem for elem in range(1,17,1)]
 
 
    msg_tag ="direct2"

    if len(send_info) > 64:
        msg_tag +=" %d"%(len(send_info))
        f = open('/home/smhong/vispark.txt','w')
        for elem in send_info:
            x, y = elem
            f.write("%s\n%s\n"%(x,y))
        f.close()
        
    else :

        msg_tag +=" 0"

        for elem in send_info:
            x,y = elem
            msg_tag += " " + x + " " + y 
    
        
    msg_tag +=" END "
    msg_tag += '0'*msgsize

    lenn = len(msg_tag)/msgsize
    msg_tag = msg_tag[:lenn*msgsize] 

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]

   
    for node in node_list:

        host_name  = node
    
        clisock = socket(AF_INET, SOCK_STREAM)
        clisock.connect((host_name,4949))

        clisock.send(head_tag+msg_tag)
        clisock.shutdown(1)
        clisock.close()
 

    #from time import sleep
    #sleep(0.02*len(send_info))

    msg_tag ="alive END"
    msg_tag += '0'*msgsize

    node_list = [] 
    node_list.append(send_info[-1][1])

    for node in node_list:

        host_name  = node
        clisock = socket(AF_INET, SOCK_STREAM)
        clisock.connect((host_name,4949))

        clisock.send(head_tag+msg_tag)
        clisock.shutdown(1)
    
        str_buf = []
        while True:
            data = clisock.recv(msgsize)
            if len(data) > 0:
                str_buf.append(data)
            else :
                break
        clisock.close()

    return "done"



 


def gpuShuffle(node_list, send_info):
    
    msg_tag ="direct"

    for elem in send_info:
        x,y,z = elem
        msg_tag += " " + x + " " + y + " " +z 
    
    msg_tag +=" END "
    msg_tag += '0'*msgsize

    lenn = len(msg_tag)/msgsize
    msg_tag = msg_tag[:lenn*msgsize] 

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
   
    for node in node_list:

        host_name  = node
        clisock = socket(AF_INET, SOCK_STREAM)
        clisock.connect((host_name,4949))

        clisock.send(head_tag+msg_tag)
        clisock.shutdown(1)
        clisock.close()
 

    msg_tag ="alive END"
    msg_tag += '0'*msgsize

    for node in node_list:

        host_name  = node
        clisock = socket(AF_INET, SOCK_STREAM)
        clisock.connect((host_name,4949))

        clisock.send(head_tag+msg_tag)
        clisock.shutdown(1)
    
        str_buf = []
        while True:
            data = clisock.recv(msgsize)
            if len(data) > 0:
                str_buf.append(data)
            else :
                break
        clisock.close()

    return "done"



def action(data,cache=False):


    host_name  = get_ib_name()
    clisock = socket(AF_INET, SOCK_STREAM)
    clisock.connect((host_name,4949))

    msg_tag ="ACT %d %s END "%(cache,"fake")
    msg_tag += '0'*msgsize
    msg_tag = msg_tag[:msgsize]
 
    lenn = (len(data) +len(msg_tag))/msgsize

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
  
    clisock.send(head_tag+data+msg_tag)
    clisock.shutdown(1)
    
    """
    msg = clisock.recv(msgsize)
    print msg

    lenn = int(msg[:msg.find('**')])
    msg = msg[msg.find('**')+2:]
    data_len = int(msg[:msg.find('**')])
    """

    #lenn = 385
    #data_len = 786432


   
    #clisock.shutdown(1)

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    #clisock.shutdown(1)
    clisock.close()

    data="".join(str_buf)

    if len(data) < msgsize:
        send_data = "HIT %s END "%(data)
        send_data += '0'*msgsize
        data = send_data[:msgsize]

    #print data

    return data

def EGLinit(node_file,width=1000,height=1000,max_vertex=8000**2,struct_size=36):

    node_list = []

    with open(node_file) as f:
        lines =f.readlines()

        for elem in lines:
            elem = elem.strip()

            if elem.find('#')==0:
                continue
            if elem.find('\n')!=-1:
                elem = elem[:-2]

            if len(elem) > 0:
                #elem = elem.replace('ib','emerald')

                node_list.append(elem)



    msg_tag ="EGLinit"

    msg_tag +=" %d %d %d %d END "%(width,height,max_vertex,struct_size)
    msg_tag += '0'*msgsize

    lenn = len(msg_tag)/msgsize
    msg_tag = msg_tag[:lenn*msgsize]

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]

    for node in node_list:

        host_name  = node
        clisock = socket(AF_INET, SOCK_STREAM)
        clisock.connect((host_name,4949))

        clisock.send(head_tag+msg_tag)
        clisock.shutdown(1)
        clisock.close()


def render(data,vert_str="",geo_str="",frag_str="",projMat=numpy.eye(4),modelMat=numpy.eye(4),viewMat=numpy.eye(4),var={}):

    msg_tag="Render "

    if len(vert_str) > 0:
        vert_len = len(vert_str)
        vert_str += '0'*(msgsize - vert_len%msgsize)
        vert_lenn = len(vert_str)/msgsize
        msg_tag += "vert %d %d "%(vert_len,vert_lenn)

    if len(geo_str) > 0:
        geo_len = len(geo_str)
        geo_str += '0'*(msgsize - geo_len%msgsize)
        geo_lenn = len(geo_str)/msgsize
        msg_tag += "geo %d %d "%(geo_len,geo_lenn)

    if len(frag_str) > 0:
        frag_len = len(frag_str)
        frag_str += '0'*(msgsize - frag_len%msgsize)
        frag_lenn = len(frag_str)/msgsize
        msg_tag += "frag %d %d "%(frag_len,frag_lenn)

    msg_tag += "shader_end "

    data_str = "" 
    if len(var) >0:
        args_info, data_str = func_dict_to_str(var)
        data_len = len(data_str)
        data_str += '0'*(msgsize - data_len%msgsize)
        lenn = len(data_str)/msgsize
        msg_tag += args_info
  
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]

    mat = projMat.astype(numpy.float32).tostring()
    mat += modelMat.astype(numpy.float32).tostring()
    mat += viewMat.astype(numpy.float32).tostring()
    mat += '0'*msgsize
    mat = mat[:msgsize]

    #print msg_tag
    #print data_str 

    return data + msg_tag + data_str + vert_str + geo_str + frag_str + mat


def extract_halo(block_idx,data,halo = 1):

    
    msg_tag="Extract %d %d "%(block_idx,halo)
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]
  
    print halo
 
    #print msg_tag
    #print data_str

    host_name  = get_ib_name()
    clisock = socket(AF_INET, SOCK_STREAM)
    clisock.connect((host_name,4949))
 
    lenn = (len(data) +len(msg_tag))/msgsize

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
  
    
 
    clisock.send(head_tag+data+msg_tag)
    clisock.shutdown(1)
    clisock.close()


    return data
    #return [args_info,data_str]


def append_halo(block_idx,data,halo = 1):

    
    msg_tag="Append %d %d "%(block_idx,halo)
    msg_tag+='0'*msgsize
    msg_tag = msg_tag[:msgsize]
   
    #print msg_tag
    #print data_str

    return data + msg_tag
    #return [args_info,data_str]




def shuffle_halo(host_name,elem_list = []):

    msg_tag="Shuffle "
    for elem in elem_list:
        idx, source, dest = elem
        msg_tag += "%d %s %s "%(idx,source,dest)
    msg_tag +="-1 "
    msg_tag +='0'*msgsize
    msg_tag = msg_tag[:msgsize]

    #print msg_tag  
 
    #print msg_tag
    #print data_str

    clisock = socket(AF_INET, SOCK_STREAM)
    clisock.connect((host_name,4949))
 
    lenn = len(msg_tag)/msgsize

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
  
    
    clisock.send(head_tag+msg_tag)
    clisock.shutdown(1)
    clisock.close()

    return 'yet'


def shuffle_check(host_name):

    msg_tag ="alive END"
    msg_tag += '0'*msgsize
    msg_tag = msg_tag[:msgsize]
    
    lenn = len(msg_tag)/msgsize

    head_tag = "Start %d END "%lenn
    head_tag += '0'*msgsize
    head_tag = head_tag[:msgsize]
 
    clisock = socket(AF_INET, SOCK_STREAM)
    clisock.connect((host_name,4949))

    clisock.send(head_tag+msg_tag)
    clisock.shutdown(1)

    str_buf = []
    while True:
        data = clisock.recv(msgsize)
        if len(data) > 0:
            str_buf.append(data)
        else :
            break
    clisock.close()

    return 'done'


class sparkHalo:


    def __init__(self):
        self.elem_dict={}
        self.node_list=[]


    def print_dict(self):
        print self.elem_dict

    def print_node(self):
        print self.node_list

    def setdata(self,rdd,local_dimx,local_dimy,local_dimz,dbytes):

        elems= rdd.map(lambda (name,data):(int(name),get_ib_name())).collect()

        self.dimx= local_dimx
        self.dimy= local_dimy
        self.dimz= local_dimz
        self.bytes = dbytes

        dicts = {}
        nodes = []

        for elem in elems:
            key,node = elem
            dicts[key] = node
            nodes.append(node) 
         
        nodes = list(set(nodes))

        self.elem_dict = dicts
        self.node_list = nodes

    def setCommModel(self,commtype=None,alltoall=False):
        
        dicts = self.elem_dict
        nodes = self.node_list

        num_elem = len(dicts)

        send_list = []

        if alltoall == True:
            for elem in dicts.keys():
                send_list.append([elem,dicts[elem],'all'])
        

        else :

            for elem in dicts.keys():

                up_idx = elem-1
                dn_idx = elem+1

                if (up_idx >= 0):
                    if dicts[up_idx] != dicts[elem]:
                        send_list.append([elem,dicts[elem],dicts[up_idx]])
                if (dn_idx < num_elem):
                    if dicts[dn_idx] != dicts[elem]:
                        send_list.append([elem,dicts[elem],dicts[dn_idx]])
       

        for elem in send_list:
            print "%s block will send from %s to %s"%(elem[0],elem[1],elem[2])
        self.send_list= send_list

    #def extract(self,rdd):
    #    rdd = rdd.map(lambda (name,data): (name,extract_halo(name,data)))

    #def append(self,rdd):
    #    rdd = rdd.map(lambda (name,data): (name,append_halo(name,data)))

    #def append(self,rdd):
    #    rdd = rdd.map(lambda (name,data): (name,append_halo(name,data)))

    def callShuffle(self):
        
        node_list = self.node_list
        send_list = self.send_list

        for node in node_list:
            shuffle_halo(node,send_list)
        
        for node in node_list:
            shuffle_check(node)


    def shuffle(self,rdd):
        #import sys
        #sys.path.append('/home/smhong/Project/vispark_example/tvcg/02.HeatFlow')

        dimx = self.dimx
        dimy = self.dimy
        dbytes = self.bytes

        #print dimx,dimy,dbytes

        ori_rdd = rdd.cache()
        rdd = rdd.map(lambda (name,data): (name,extract_halo(name,data,dimx*dimy*dbytes)))
        #rdd = rdd.groupByKey().mapValues(list).map(lambda (name,data): (name,data[0])).cache()

        rdd.foreach(lambda (name,data):(int(name)))
        self.callShuffle()
        rdd = ori_rdd.map(lambda (name,data): (name,append_halo(name,data,dimx*dimy*dbytes)))

        return rdd


