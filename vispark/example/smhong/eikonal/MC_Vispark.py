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

def init_patch(name,data,dim,num_block,profiler=None):
   
    #profiler.start("Read") 
    #idx = int(name[name.rfind('_')+1:])
    idx = name

    data = numpy.zeros(dim)
    #data = numpy.fromstring(data,dtype=numpy.float32)

    data.fill(10000)

    #print idx, data[4000000:4000100] , raw[4000000*4:4000100*4] 
    data = data.reshape(dim)

    #if idx == num_block/2:
    #    data[dim[0]/2][dim[1]/2][dim[2]/2] =0

    for i in range(dim[0]):
        data[i][dim[1]/2][dim[2]/2] =0
    

    #data = numpy.fromstring(data,dtype=numpy.float32).reshape(dim)
    
    #data = data.astype(numpy.float32)

    newdata = numpy.zeros((dim[0]+2,dim[1],dim[2]),dtype=numpy.float32)
    newdata.fill(10000)
    newdata[1:1+dim[0],:,:] = data

    #print newdata.max() 

    #scr_name="screen_00"       
    #profiler.stop("Read") 
    #print_time("init_patch_2_%d"%(idx))
 
    return (idx,newdata) 
    #return (scr_name,(name,data)) 
    #return (name,numpy.array(range(10)))




if __name__ == "__main__":

    import sys
    
    width = 1000
    height = 1000


    dataType =0
 
    if len(sys.argv) > 1:
        dataType = int(sys.argv[1])

    width = 1000
    height = 1000

    ImgSplit = [16,1,1]
    ImgDim = [1344,1344,1344]

    if dataType == 0: 
        ImgSplit = [16,1,1]
        ImgDim = [1344,1344,1344]
        #ImgDim = [256,256,256]
    elif dataType == 1: 
        ImgSplit = [16,1,1]
        ImgDim = [1696,1696,1696]
    elif dataType == 2: 
        ImgSplit = [32,1,1]
        ImgDim = [2144,2144,2144]
    elif dataType == 3: 
        ImgSplit = [64,1,1]
        ImgDim = [2688,2688,2688]

    ImageName = "eikonal_%d_%d"%(ImgDim[0],ImgSplit[0])
    
    numSplit = ImgSplit[0]

    gpuinit("/home/smhong/Project/vispark/conf/slaves")
    #EGLinit("/home/smhong/Project/vispark/conf/slaves",width,height,max_vertex=120**2,struct_size=24)
    EGLinit("/home/smhong/Project/vispark/conf/slaves",width,height,max_vertex=120**2,struct_size=36)

    dummy = []
    for i in range(numSplit):
        dummy.append([i,0])

    print dummy

    sc = SparkContext(appName="Eikonal_%s"%(ImageName))
    sc.addFile("/home/smhong/vispark_hvcl/import/visAPIs.py")

    pixels = sc.parallelize(dummy,numSplit)


    print ImgDim
    print ImgSplit

    LocalDim = map(lambda x,y:x/y,ImgDim,ImgSplit)    

    print LocalDim

    dimx = LocalDim[2]
    dimy = LocalDim[1]
    dimz = LocalDim[0]




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

    print matrix


    rdd = pixels.map(lambda (name,data): init_patch(name,data,LocalDim,numSplit))
    
    rdd = rdd.map(lambda (name,data): (name,send_new(data)))
    
    if flag_GPUcache :
        rdd = rdd.map(lambda (name,data): (name,action(data,True)))
   
    rdd = rdd.cache()

    node_list = getnodelist("/home/smhong/Project/vispark/conf/slaves")

    print node_list

    #from sparkassisy import sparkHalo 
    
    sparkHalo = sparkHalo()

    sparkHalo.setdata(rdd,dimx,dimy,dimz+2,4)

    sparkHalo.print_dict()
    sparkHalo.print_node()

    sparkHalo.setCommModel('z_only',False)
    #sparkHalo.setCommModel('z_only',False)

    fake_arg = numpy.zeros(5).astype(numpy.int32)
    iso_label = numpy.array([917,1000]).astype(numpy.int32)


    doRender = True
    doShuffle = True
    doViewer = True 
   
    if doViewer:
        func_name = "mc_center"
        max_iter = 5000
    else :
        func_name = "mc_char"
        max_iter = 5
 
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

    phong_vert = open("phong.vert").read()




    vardict={}
    #vardict["ambient_color"] = numpy.array([1.0,0.0,0.0,0.0]).astype(numpy.float32)
    #vardict["diffuse_color"] = numpy.array([0.0,1.0,0.0,0.0]).astype(numpy.float32)

    vardict["ambient_color"] = numpy.array([0.6,0.6,1.0,0.0]).astype(numpy.float32)
    vardict["diffuse_color"] = numpy.array([0.8,0.8,1.0,0.0]).astype(numpy.float32)
    #vardict["ambient_color"] = numpy.array([1.0,0.5,0.2,0.0]).astype(numpy.float32)
    #  vardict["diffuse_color"] = numpy.array([1.0,0.2,0.5,0.0]).astype(numpy.float32)
   #
    ptx_code = open('mc.ptx','r').read()
  
    max_iter =1

    iter_cnt = 0 
    while True :
        if doViewer: 
            #mmtx,inv,ray_direction,max_iter,iso_label,color_data= recv_mmtx_info(sersock)
            mmtx,inv,ray_direction,max_iter,num_label,iso_label,color_table= recv_mmtx_info(sersock)
            #mmtx,inv,ray_direction,v1,v2,v3,v4 = recv_mmtx_info(sersock)

            #num_label = 2
            print max_iter,num_label,iso_label,color_table

        for i in range(max_iter):
       
            iter_cnt +=1
            t1 = time.time()

            if doShuffle :
                rdd = sparkHalo.shuffle(rdd)
                rdd = rdd.cache()

            t2 = time.time()
        #print data 
        
            rdd= rdd.map(lambda (name,data): (name,run(data,ptx_code,"eikonal_kernel",[name,dimz,dimx,dimy,dimz,fake_arg,numSplit,3],(dimx*dimy*(dimz+2)*4),[dimx,dimy,dimz],True)))

            rdd = rdd.map(lambda (name,data):(name,action(data,True)))
            rdd = rdd.cache()
        
        if doRender:
            phong_frag = open("phong.frag").read()
            #iso_label = numpy.array([iso_label,42]).astype(numpy.float32)
            data = rdd.map(lambda (name,data): (name,run(data,ptx_code,"mc_kernel",[name,dimz,dimx,dimy,dimz,fake_arg,iso_label,0,num_label,ImgDim[0],color_table],(2000*1000*1000),[dimx,dimy,dimz+2])))
            if doViewer:
                data = data.map(lambda (name,data): (name,render(data,vert_str=phong_vert,frag_str=phong_frag,modelMat=mmtx,var=vardict)))
            else :
                data = data.map(lambda (name,data): (name,render(data,vert_str=phong_vert,frag_str=phong_frag,projMat=projMat,modelMat=modelMat,viewMat=viewMat,var=vardict)))
            #data = data.map(lambda (name,data): (name,render(data,vert_str=phong_vert,frag_str=phong_frag,modelMat=viewMat,viewMat=mmtx,var=vardict)))

            data = data.map(lambda (name,data):(name,action(data)))


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
        else :
            result = rdd.collect() 
        t5 = time.time()
        print "\nIter : ",iter_cnt
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


