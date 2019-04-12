from pyspark import SparkContext, SparkConf

import sys
sys.path.append('/home/smhong/vispark_hvcl/import/')
from visAPIs import *

import numpy
from PIL import Image
import sys
import getpass
from hdfs import InsecureClient
import time
from socket import *



def init_patch(name,data,dim,halo=0,profiler=None):
   
    #profiler.start("Read") 
    idx = int(name[name.rfind('_')+1:])

    #data = numpy.fromstring(data,dtype=numpy.uint8).reshape((dim[0],dim[1],dim[2]))
    data = numpy.fromstring(data,dtype=numpy.float32).reshape(tuple(dim)+(11,))
    
    newdata = numpy.zeros((dim[0]+2*halo,dim[1]+2*halo,dim[2]+2*halo,11),dtype=numpy.float32)
    
    newdata[halo:halo+dim[0],halo:halo+dim[1],halo:halo+dim[2],:] = data
   
    #print data[:,:,:,2].min() , data[:,:,:,2].max() 
    #data = data.astype(numpy.float32)

    #scr_name="screen_00"       
    #profiler.stop("Read") 
 
    return (name,newdata) 
    #return (scr_name,(name,data)) 
    #return (name,numpy.array(range(10)))

def get_data_range(name,data_shape,split_shape,halo=0):
   
    #halo = 0
    data = numpy.zeros(26).astype(numpy.int32) 

    idx = int(name[name.rfind('_')+1:])
    block_num = reduce(lambda x,y:x*y,split_shape)
    
    local_data_shape = map(lambda x,y:x/y,data_shape,split_shape)    
    
    start=[0,0,0]
    end=local_data_shape[::-1] 

    nx = split_shape[2]
    ny = split_shape[1]

    z = idx/(nx*ny)
    y = (idx/nx)%ny
    x = idx%nx

    start[0] += x*local_data_shape[2] 
    end  [0] += x*local_data_shape[2] 
    start[1] += y*local_data_shape[1] 
    end  [1] += y*local_data_shape[1] 
    start[2] += z*local_data_shape[0] 
    end  [2] += z*local_data_shape[0] 


    data[:3] = start
    data[4:7] = end
    data[8:11] = [0,0,0]
    data[12:15] = data_shape[::-1]
    data[16:19] = start
    data[20:23] = end
  
    data[0] -= halo
    data[1] -= halo
    data[2] -= halo
    data[4] += halo
    data[5] += halo
    data[6] += halo
    data[16] -= halo
    data[17] -= halo
    data[18] -= halo
    data[20] += halo
    data[21] += halo
    data[22] += halo


    data_shape = numpy.array(data_shape[::-1])

    data[:3]    -= data_shape/2
    data[4:7]   -= data_shape/2 
    data[8:11]  -= data_shape/2
    data[12:15] -= data_shape/2
    data[16:19] -= data_shape/2
    data[20:23] -= data_shape/2 
    #data[24] = halo
 
    return data

def get_z_idx(name,num,data_shape,split_shape,mmtx):
       
   # import random 
    #return  int(name[name.rfind('_')+1:])

    idx = int(name[name.rfind('_')+1:])

    local_data_shape = map(lambda x,y:x/y,data_shape,split_shape)    
    
    start=[0,0,0]
    end=local_data_shape[::-1] 

    nx = split_shape[2]
    ny = split_shape[1]

    z = idx/(nx*ny)
    y = (idx/nx)%ny
    x = idx%nx

    start[0] += x*local_data_shape[2] 
    end  [0] += x*local_data_shape[2] 
    start[1] += y*local_data_shape[1] 
    end  [1] += y*local_data_shape[1] 
    start[2] += z*local_data_shape[0] 
    end  [2] += z*local_data_shape[0] 
 

    def make_depth(mmtx,start,end):
        x,y,z,depth = 0, 0, 0, 0
        
        x = (start[0] + end[0])/2
        y = (start[1] + end[1])/2
        z = (start[2] + end[2])/2


        #X = mmtx[0][0]*x+mmtx[0][1]*y+mmtx[0][2]*z+mmtx[0][3]
        #Y = mmtx[1][0]*x+mmtx[1][1]*y+mmtx[1][2]*z+mmtx[1][3]
        #Z = mmtx[2][0]*x+mmtx[2][1]*y+mmtx[2][2]*z+mmtx[2][3]
        Z = mmtx[2][0]*x+mmtx[2][1]*y+mmtx[2][2]*z
        #Z = mmtx[0][2]*x+mmtx[1][2]*y+mmtx[2][2]*z+mmtx[3][2]
        return Z
        #print idx,x,y,z,Z
        #depth = Z*Z
        #import math
        #return math.sqrt(depth)

    return (make_depth(mmtx,start,end), start, end, name)



def get_first_key(name,block_list):

    num_image = len(block_list)
    idx = block_list.index(name)

    return idx

def union(x,y):

    print x[0],x[1][:20]
    print y[0],y[1][:20]

    array = []

    if x[0] < y[0]:
        array.append(x[1])        
        array.append(y[1])        
    else :
        array.append(y[1])        
        array.append(x[1])        

 
    return array



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


def Rotate(angle, x, y, z, modelview, inv_modelview, flag=False):
    import math
    pi = math.pi


    l = x*x + y*y + z*z
    l = 1/math.sqrt(l)
    x = x*l
    y = y*l
    z = z*l

    #matrix
    th = math.pi/180*(angle)
    c = math.cos(th)
    s = math.sin(th)
    tm = numpy.array([ x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0, x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0, x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,0, 0,0,0,1], dtype=numpy.float32)
    tm = tm.reshape((4,4))
    #print modelview
    if flag==True:
        modelview = numpy.dot(tm, modelview)
    else:
        modelview = numpy.dot(modelview, tm)


    #inverse
    th = math.pi/180*(-angle)
    c = math.cos(th)
    s = math.sin(th)
    tm = numpy.array([ x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0, x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0, x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,0, 0,0,0,1], dtype=numpy.float32)
    tm = tm.reshape((4,4))
    if flag == True:
        inv_modelview = numpy.dot(inv_modelview, tm)
    else:
        inv_modelview = numpy.dot(tm, inv_modelview)

    return modelview, inv_modelview

def Translate(x, y, z, modelview, inv_modelview, flag = False):
    #matrix
    tm = numpy.eye(4,dtype=numpy.float32)
    tm[0][3] = x
    tm[1][3] = y
    tm[2][3] = z
    
    if flag==True:
        modelview = numpy.dot(tm, modelview)
    else:
        modelview = numpy.dot(modelview, tm)

    #inverse matrix
    tm = numpy.eye(4,dtype=numpy.float32)
    tm[0][3] = -x
    tm[1][3] = -y
    tm[2][3] = -z
    if flag == True:
        inv_modelview = numpy.dot(inv_modelview, tm)
    else:
        inv_modelview = numpy.dot(tm, inv_modelview)

    return modelview, inv_modelview


def Scale(x, y, z, modelview, inv_modelview, flag = False):
    #matrix
    tm = numpy.eye(4,dtype=numpy.float32)
    tm[0][0] = x
    tm[1][1] = y
    tm[2][2] = z
    
    if flag==True:
        modelview = numpy.dot(tm, modelview)
    else:
        modelview = numpy.dot(modelview, tm)

    #inverse matrix
    tm = numpy.eye(4,dtype=numpy.float32)
    tm[0][0] = 1.0/x
    tm[1][1] = 1.0/y
    tm[2][2] = 1.0/z

    if flag == True:
        inv_modelview = numpy.dot(inv_modelview, tm)
    else:
        inv_modelview = numpy.dot(tm, inv_modelview)

    return modelview, inv_modelview






if __name__ == "__main__":

    username = 'smhong'
    
    outType = 2
    dataType = 5
    flag_GPUcache = True
    flag_GPUshuffle = True
    ImageNums = 10
    base_level = 4
    halo =1
 
    if len(sys.argv) > 1:
        dataType = int(sys.argv[1])
    if len(sys.argv) > 2:
        outType  = int(sys.argv[2])
    if len(sys.argv) > 3:
        flag_GPUcache = int(sys.argv[3])
    if len(sys.argv) > 4:
        flag_GPUshuffle = int(sys.argv[4]) 
    if len(sys.argv) > 5:
        ImageNums = int(sys.argv[5]) 
 
    if dataType == 1:
        ImageName = 'water_840_520_448_16'
    elif dataType == 2:
        ImageName = 'water_1064_652_560_16'
    elif dataType == 3:
        ImageName = 'water_1344_824_704_32'
    elif dataType == 4:
        ImageName = 'water_1696_1036_888_64'
    elif dataType == 5:
        ImageName = 'water_2136_1304_1116_128'
        
    #ImageName = 'water_1d_1344_824_704_16'
    #ImageName = 'water_1d_1064_652_560_16'
    #ImageName = 'water_1696_1036_888_64'

    #ImageName = 'water1_420_260_224_4'
    ImageName = 'water0_1064_652_560_16'

    ImagePath = 'hdfs://emerald:9000/user/' + username + '/' + ImageName + '/'
    
    ImgDim = [-1,-1,-1]
    ImgSplit = [1,1,1]

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



    dimx = 500
    dimy = dimx


    if   outType == 1:    
        dimx = dimx*1 
        dimy = dimy*1 
    elif outType == 2:   
        dimx = dimx*2 
        dimy = dimy*2 
    elif outType == 3:   
        dimx = dimx*4
        dimy = dimy*4 
    elif outType == 4:   
        dimx = dimx*8
        dimy = dimy*8 
    elif outType == 5:   
        dimx = dimx*16
        dimy = dimy*16



    print "Total Size" , ImgDim
    print "Screen Size", dimx, dimy
    print "GPU cache", flag_GPUcache
    print "GPU shuffle",flag_GPUshuffle 
    

    xoff, yoff = 0, 0
    xoff = -dimx/2
    yoff = -dimy/2


    def read_transfer():

        mat = open('transfer/color_table_water_210_130_112_mat.txt').readlines()
        prs = open('transfer/color_table_water_210_130_112_prs.txt').readlines()
        rho = open('transfer/color_table_water_210_130_112_rho.txt').readlines()
        snd = open('transfer/color_table_water_210_130_112_snd.txt').readlines()
        tev = open('transfer/color_table_water_210_130_112_tev.txt').readlines()
        v02 = open('transfer/color_table_water_210_130_112_v02.txt').readlines()
        v03 = open('transfer/color_table_water_210_130_112_v03.txt').readlines()
         

        def get_array(data):
            tmp = []

            for elem in data:
                tmp.append(elem[:-1].split(' '))
                    
            return numpy.array(tmp).astype(numpy.float32)

        mat = get_array(mat)
        prs = get_array(prs)
        rho = get_array(rho)
        snd = get_array(snd)
        tev = get_array(tev)
        v02 = get_array(v02)
        v03 = get_array(v03)

        print rho
        return numpy.array([mat,rho,tev,prs,snd,v02,v03,v03,v03,v03,v03])
        #return rho

    #transfer_func = read_transfer().reshape(256*4*7)
    transfer_func = read_transfer().reshape(256*4*11)
    
    model= numpy.identity(4).astype(numpy.float32)        
    inv_model = numpy.identity(4).astype(numpy.float32)        
    empty_data= numpy.zeros((5,5)).astype(numpy.float32)

   

    gpuinit("/home/smhong/Project/vispark/conf/slaves")
    node_list = getnodelist("/home/smhong/Project/vispark/conf/slaves")
    print node_list
    time.sleep(5)

    #  sparkHalo = sparkHalo()
    #
    #  sparkHalo.setdata(rdd,dimx,dimy,dimz+2,4*11)
    #
    #  sparkHalo.print_dict()
    #  sparkHalo.print_node()
    #
    #  sparkHalo.setCommModel('z_only',False)
    #

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
    
    
    t0 = time.time()    
 
    sc = SparkContext(appName="composite_%s_[%dx%d]"%(ImageName,dimx,dimy),environment={"spark.executor.cores":str(ImageNums/16)})

    sc.addFile("/home/smhong/vispark_hvcl/import/visAPIs.py")

    ImagePatch = sc.binaryFiles(ImagePath,ImageNums*2)
    

    ImagePatch = ImagePatch.map(lambda (name,data): init_patch(name,data,LocalDim,halo))

    ImagePatch = ImagePatch.map(lambda (name,data): (name,send_new(data)))
    
    if flag_GPUcache :
        ImagePatch = ImagePatch.map(lambda (name,data): (name,action(data,True)))
            
    ImagePatch = ImagePatch.cache()

    ImageList = ImagePatch.map(lambda (name,data): (name)).collect()
    
    #ImagePatch = sparkHalo.shuffle(ImagePatch)
    t1 = time.time()    
   
    print "Loading Time : ",t1-t0
 
    import math
    print "%s [%d,%d]"%(ImageName,dimx,dimy)

    i =0
    #for i in range(3):
    while True:
        print "Screen ",i


        mmtx,inv,ray_direction,transfer_func = recv_mmtx_info(sersock,transfer_func)

        f = open("transfer.txt","w")
        f.write(transfer_func.tostring())
        f.close()


        f1 = open("rotate.txt","a")
        f1.write(mmtx.tostring())
        f1.write(inv.tostring())
        f1.write(ray_direction.tostring())
        f1.close()
        ptx_code = open('ray_casting.ptx','r').read()
        try :

            t_list = []
            t_list.append(time.time())
            
            z_idx_list = map(lambda (name) : (get_z_idx(name,i,ImgDim,ImgSplit,mmtx)),ImageList)
        
            z_idx_list = sorted(z_idx_list,reverse = True)

            sorted_block_list= []

            for elem in z_idx_list:
                sorted_block_list.append(elem[3])




            imgs = ImagePatch.map(lambda (name,data): (get_first_key(name,sorted_block_list),run(data,ptx_code,"render",[get_data_range(name,ImgDim,ImgSplit,halo),dimx,dimy,transfer_func,inv,xoff,yoff,11,ray_direction],dimx*dimy*4*4,[dimx,dimy])))


            if flag_GPUshuffle:
                imgs = imgs.map(lambda (name,data):(name,action(data)))
            else :
                imgs = imgs.map(lambda (name,data):(name,recv(data)))

            group_level = []
            block_num = len(sorted_block_list)
       
            cur_level = base_level 
            while cur_level <= block_num:
                group_level.append(base_level)
                cur_level *= base_level 

            sum_group_level = reduce(lambda x,y:x*y,group_level)

            if block_num/sum_group_level > 1:
                group_level.append(block_num/sum_group_level)

            #print group_level 

            tt0 = 0
            #for ll in range(len(group_level)): 
            #for ll in range(len(group_level)):
            cur_level = 0 
            for level in group_level:
                #print "Level %d"%level
                isFinal   = False
                if cur_level == len(group_level)-1:
                    isFinal = True
                cur_level += 1 

                def sort_image(data):
                    data = sorted(data)
    
                    arr = []

                    for elem in data:
                        arr.append(elem[1]) 

                    return arr               

    
                #imgs = imgs.map(lambda(idx,data): (idx/2,(idx,data))).reduceByKey(union)
                imgs = imgs.map(lambda(idx,data): (idx/level,(idx,data))).groupByKey().mapValues(list)
                imgs = imgs.map(lambda(idx,data): (idx,sort_image(data)))
                #imgs = imgs.map(lambda(idx,data): (idx/2,(idx,data))).groupByKey().mapValues(list)
        
 
                def arrange_after(after):
            
                    new_after =[]

                    for elem in after:
                        for data in elem:
                            new_after.append(data)
    
                    return sorted(new_after)


                if flag_GPUshuffle :
            
                    imgs = imgs.cache()

                    after = imgs.map(lambda(idx,data):(getOnlyTag(data))).collect()
   
                    #print "Original" ,after
 
                    after = arrange_after(after)
   
                    #print "Arrange", after
 
                    t_list.append(time.time())
     

                    tt1 = time.time()
                    newShuffle(after,node_list)
                    tt2 = time.time()

                    tt0 += tt2-tt1
                    t_list.append(time.time())
            

                print cur_level, level
                func_args = [level,dimx,dimy]

                if flag_GPUshuffle:


                    imgs = imgs.map(lambda (name,data): (name,send_seq2(data)))
                    if isFinal == True:
                        #imgs = imgs.map(lambda (name,data): (name,run(data,ptx_code,"composite_uchar",[level,dimx,dimy],dimx*dimy*3,[dimx,dimy])))
                        imgs = imgs.map(lambda (name,data): (name,run(data,ptx_code,"composite_uchar",func_args,dimx*dimy*3,[dimx,dimy])))
                        imgs = imgs.map(lambda (name,data):(name,viewer(data,dimx*dimy*3)))
                        #imgs = imgs.map(lambda (name,data):(name,saveFile(data,"/home/smhong/screen_%02d.raw"%i,dimx*dimy*3)))
                        #imgs = imgs.map(lambda (name,data):(name,action(data)))
                    else:
                        imgs = imgs.map(lambda (name,data): (name,run(data,ptx_code,"composite",func_args,dimx*dimy*4*4,[dimx,dimy])))
                        imgs = imgs.map(lambda (name,data):(name,action(data)))


                    #else :
                else :
                    imgs = imgs.map(lambda (name,data): (name,send_seq(data)))
    
                    if isFinal == True:
                        imgs = imgs.map(lambda (name,data): (name,run(data,ptx_code,"composite_uchar",func_args,dimx*dimy*3,[dimx,dimy])))
                        #imgs = imgs.map(lambda (name,data):(name,saveFile(data,"/home/smhong/screen_%02d.raw"%i,dimx*dimy*3)))
                        #imgs = imgs.map(lambda (name,data):(name,saveFile(data,"/home/smhong/screen_%02d.raw"%i,dimx*dimy*3)))
                        #imgs = imgs.map(lambda (name,data):(name,action(data)))
                        imgs = imgs.map(lambda (name,data):(name,viewer(data,dimx*dimy*3)))
        
                    else : 
                        imgs = imgs.map(lambda (name,data): (name,run(data,ptx_code,"composite",func_args,dimx*dimy*4*4,[dimx,dimy])))
                        imgs = imgs.map(lambda (name,data):(name,recv(data)))


 
            data = imgs.collect()

            t3 = time.time()
            t_list.append(time.time())
       
            #print t_list, len(t_list)
            #  print "MPI                 :",t_list[2] - t_list[1]
            #  print "Composite + shuffle :",t_list[3] - t_list[2]
            #  print "MPI                 :",t_list[4] - t_list[3]
            #  print "Composite + shuffle :",t_list[5] - t_list[4]
            #  print "MPI                 :",t_list[6] - t_list[5]
            #  print "Composite + shuffle :",t_list[7] - t_list[6]
            #  print "MPI                 :",t_list[8] - t_list[7]
            #
            print "\nTotal : ",t_list[-1] - t_list[0]
            i += 1
            #  for j in range(1,len(t_list)):
            #      if j == 1:
            #          print "Render   + shuffle  :",t_list[j] - t_list[j-1]
            #      elif j == len(t_list)-1:
            #          print "Composite + Save    :",t_list[j] - t_list[j-1]
            #      elif j % 2 == 0:
            #          print "MPI                 :",t_list[j] - t_list[j-1]
            #      else :
            #          print "Composite + shuffle :",t_list[j] - t_list[j-1]
            #
            #  for j in range(1,len(t_list)):
            #    print "%f "%(t_list[j]-t_list[j-1]),
            #  print "\n"
        except error as msg:
            print "Rendering Fail " + str(msg[0]) + " - " + str(msg[1])


    #profiler.close()
    sc.stop()
    
    #profiler.draw() 
    #print "\n\n\n"

    gpuinit("/home/smhong/Project/vispark/conf/slaves")
    time.sleep(5)

