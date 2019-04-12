# VISPARK

from pyspark import SparkContext
import numpy, math
from PIL import Image
import collections
from socket import *
import getpass
from hdfs import InsecureClient
import time

import sys
sys.path.append('/home/smhong/vispark_hvcl/import/')
from visAPIs import *



def getOnlyTag(data):
  
    t1 = time.time() 
    tags = []

    host_name = get_ib_name()
 
    if isinstance(data,list) == False:
        tags.append([data[4:16],host_name])

    else :
        for elem in data:
            tags.append([elem[4:16],host_name])
    
    t2 = time.time() 

    print t2-t1
    return tags



def init_patch(name,data,dim,profiler=None):
   
    #profiler.start("Read") 
    idx = int(name[name.rfind('_')+1:])

    #print_time("init_patch_1_%d"%(idx))
    #data = numpy.fromstring(data,dtype=numpy.uint8).reshape((dim[0],dim[1],dim[2]))
    data = numpy.fromstring(data,dtype=numpy.uint8).reshape(dim)
    newdata = numpy.zeros((dim[0]+2,dim[1],dim[2]),dtype=numpy.uint8)

    newdata[1:1+dim[0],:,:] = data


    #scr_name="screen_00"       
    #profiler.stop("Read") 
 
    return (idx,newdata) 
    #return (scr_name,(name,data)) 
    #return (name,numpy.array(range(10)))


if __name__ == "__main__":

    import sys


    sigma = 1.0

    dataType =2
    splitType = 4
    postfix = "new"
 
    if len(sys.argv) > 1:
        dataType = int(sys.argv[1])

    width = 1000
    height = 1000

    if dataType == 0: 
        dim_value = 1664
    elif dataType == 1: 
        dim_value = 2048
    elif dataType == 2: 
        dim_value = 2560
    elif dataType == 3: 
        dim_value = 3200
        postfix = "16"
    else :
        dim_value = 1664
    
    #ImageName = "mandel_%d_%s"%(dim_value,postfix);
    ImageName = "mandel_%d_new"%(dim_value)

    #ImageName = "mandel_512"
    #dim_value = 512
    #
    gpuinit("/home/smhong/Project/vispark/conf/slaves")
    #EGLinit("/home/smhong/Project/vispark/conf/slaves",width,height,max_vertex=120**2,struct_size=24)
    EGLinit("/home/smhong/Project/vispark/conf/slaves",width,height,max_vertex=120**2,struct_size=24)


    sc = SparkContext(appName="MarchingCube_Vispark_%s"%(ImageName))
    sc.addFile("/home/smhong/vispark_hvcl/import/visAPIs.py")
    print"pyspark version:" + str(sc.version)

    ImgDim = [-1,-1,-1]
    ImgSplit = [1,1,1]

    username = getpass.getuser()
    client = InsecureClient('http://emerald:50070',user=username)
    with client.read(ImageName +'/.meta', encoding='utf-8') as reader:
        content = reader.read().split('\n')
        
        for elem in content:
            if elem.startswith('X : '):
                ImgDim[2] = int(elem[4:])
            if elem.startswith('Y : '):
                ImgDim[1] = int(elem[4:])
            if elem.startswith('Z : '):
                ImgDim[0] = int(elem[4:])
            if elem.startswith('X split : '):
                ImgSplit[2] = int(elem[10:])
            if elem.startswith('Y split : '):
                ImgSplit[1] = int(elem[10:])
            if elem.startswith('Z split : '):
                ImgSplit[0] = int(elem[10:])

    print ImgDim
    print ImgSplit

    LocalDim = map(lambda x,y:x/y,ImgDim,ImgSplit)    

    print LocalDim

    dimx = LocalDim[2]
    dimy = LocalDim[1]
    dimz = LocalDim[0]


    ptx_code = open('mc_char.ptx','r').read()

    #Read Data from HDFS
    pixels = sc.binaryFiles('hdfs://emerald:9000/user/smhong/%s'%ImageName)

    #Data passing and Save in GPU 

    flag_GPUcache = True
    flag_GPUshuffle =True

    model= numpy.identity(4).astype(numpy.float32)        
    inv_model = numpy.identity(4).astype(numpy.float32)       

    mat_file = open('/home/smhong/mandel.txt').readlines()
    #mat_file = open('cell_seg.txt').readlines()
    matrix = []
    for elem in mat_file:
        matrix.append(elem.split(" ")[:-1])
    matrix = numpy.array(matrix).astype(numpy.float32)
    projMat = matrix[:4]
    modelMat = matrix[4:8]
    viewMat = matrix[8:]

    #viewMat = numpy.eye(4).astype(numpy.float32)
    print matrix


    rdd = pixels.map(lambda (name,data): init_patch(name,data,LocalDim))
    
    rdd = rdd.map(lambda (name,data): (name,send_new(data)))
    
    if flag_GPUcache :
        rdd = rdd.map(lambda (name,data): (name,action(data,True)))
   
    rdd = rdd.cache()

    node_list = getnodelist("/home/smhong/Project/vispark/conf/slaves")

    print node_list

    #from sparkassisy import sparkHalo 
    
    sparkHalo = sparkHalo()

    sparkHalo.setdata(rdd,dimx,dimy,dimz+2,1)

    sparkHalo.print_dict()
    sparkHalo.print_node()

    sparkHalo.setCommModel('z_only',False)
    #sparkHalo.setCommModel('z_only',False)

    fake_arg = numpy.zeros(5).astype(numpy.int32)
    #iso_label = iso_label[:5]
    iso_label = numpy.array([917,1000]).astype(numpy.int32)
    #iso_label = numpy.array([917,1180,2226,2251,9151]).astype(numpy.int32)
    #print iso_label 

    
    #print len(phong_vert), len(phong_frag)

    doRender = True
    doShuffle = True
    doViewer = True 
   
    if doViewer:
        func_name = "mc_center"
        max_iter = 5000
    else :
        func_name = "mc_char"
        max_iter = 5
 
    if doShuffle :
        rdd = sparkHalo.shuffle(rdd)
        rdd = rdd.cache()

    HOST =''
    PORT =5957
    sersock = socket(AF_INET,SOCK_STREAM)
    print "Socket Created", sersock
    try :
        sersock.bind((HOST,PORT))
    except error as msg:
        print "Bind failed " + str(msg[0]) + " - " + str(msg[1])
        sys.exit()

    print "Socket Bind"
    sersock.listen(10)



    vardict={}
    #vardict["ambient_color"] = numpy.array([1.0,0.0,0.0,0.0]).astype(numpy.float32)
    #vardict["diffuse_color"] = numpy.array([0.0,1.0,0.0,0.0]).astype(numpy.float32)
    
    vardict["ambient_color"] = numpy.array([0.6,0.6,1.0,0.0]).astype(numpy.float32)
    vardict["diffuse_color"] = numpy.array([0.8,0.8,1.0,0.0]).astype(numpy.float32)
    
    
  
    for i in range(max_iter):
       
        if doViewer: 
            mmtx,inv,ray_direction,phong_vert,phong_frag= recv_mmtx_info(sersock)
        
        #  print "============="
        #  print phong_vert
        #  print "============="
        #  print phong_frag
        #  print "============="
        #phong_vert = open("phong.vert").read()
        #phong_frag = open("phong.frag").read()
        #print mmtx,inv
        #print rdd
        #data = data.map(lambda (name,data): (name,sparkHalo.extract(name,data)))
        t1 = time.time()


        t2 = time.time()
        #print data 
        data = rdd.map(lambda (name,data): (name,run(data,ptx_code,func_name,[name,dimz,dimx,dimy,dimz,fake_arg,iso_label,0,1,dim_value],(8000*1000*1000),[dimx,dimy,dimz+2])))
        
        if doRender:
            if doViewer:
                data = data.map(lambda (name,data): (name,render(data,vert_str=phong_vert,frag_str=phong_frag,modelMat=mmtx,var=vardict)))
            else :
                data = data.map(lambda (name,data): (name,render(data,vert_str=phong_vert,frag_str=phong_frag,projMat=projMat,modelMat=modelMat,viewMat=viewMat,var=vardict)))
            #data = data.map(lambda (name,data): (name,render(data,vert_str=phong_vert,frag_str=phong_frag,modelMat=viewMat,viewMat=mmtx,var=vardict)))

        if flag_GPUshuffle:
            data = data.map(lambda (name,data):(name,action(data)))
        else :
            data = data.map(lambda (name,data):(name,recv(data)))


        if doRender:
            data = data.map(lambda(idx,data): ("mc_%02d"%i,data)).groupByKey().mapValues(list)

            data = data.cache()
            after = data.map(lambda(idx,data):(getOnlyTag(data))).collect()
            t3 = time.time()
           
            #print after[0] 
            #after = arrange_after(after)
            
            #print after[0]
            newShuffle(after[0],node_list)
            t4 = time.time()


            data = data.map(lambda (name,data): (name,send_seq2(data)))
            data = data.map(lambda (name,data): (name,run(data,ptx_code,"composite",[16,width,height,1,1,fake_arg],width*height*7,[width,height])))
            #data = data.map(lambda (name,data):(name,action(data)))
            if doViewer: 
                data = data.map(lambda (name,data):(name,viewer(data,width*height*3)))
            else :
                data = data.map(lambda (name,data):(name,saveFile(data,"/home/smhong/mc_%02d.raw"%i,width*height*3)))
                data = data.map(lambda (name,data):(name,action(data)))

        
        result = data.collect()
        t5 = time.time()
        print "\nIter : ",i
        print "Total : ",t5-t1
        print "Shuffle : ",t2-t1
        if doRender:
            print "Spark collect:",t3-t2
            print "MPI shuffle: ",t4-t3
            print "Composite Save: ",t5-t4
        #print time.time()

       
    if flag_GPUshuffle == True:
        result =[]

    for elem in result:
        name, value = elem
        arr = numpy.fromstring(value,dtype=numpy.float32)
        print arr[:21]
        #arr = arr[0:-1,1:-1,1:-1]
        #f = open("block_%d.raw"%name,"w")
        #f.write(arr.tostring())
        #f.close()
            #    print type(value), len(value)


    #profiler.close()
    sc.stop()
    
    #profiler.draw() 
    print "\n\n\n"

    #gpukill("/home/smhong/Project/vispark/conf/slaves")
    gpuinit("/home/smhong/Project/vispark/conf/slaves")
    #time.sleep(5)


