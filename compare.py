import numpy as np
import xlsxwriter
from decimal import *




'''
    Description: 
        wrapper for the xlsxwriter workbook creation function so you dont have to remember it everytime
    Inputs:
        name - string name of the excel file to be created
    Outputs: 
        returns an xlsxwriter workbook for you to use in the other functions
'''
#creates the workbook that needs to be passed to compare_xlsxwriter (just a wrapper for an xlsxwriter function)
def create_workbook(name):
    return xlsxwriter.Workbook(name)
 



'''
    Description: 
        compare blobs using xlsxwriter and save in excel file
    Inputs:
        workbook - an xlsxwriter workbook which is used to write the information to excel files
        caffetxt - string name of file where the caffe layer's output is stored
        cpptxt - string name of file where the cpp layer's output is stored
        caffeSheetName - string of desired name for the caffe sheet in the excel file. Useful for differentiating multiple layers in a single file. If nothing is provided will default to Sheet1, Sheet2, etc.
        cppSheetName - string of desired name for the cpp sheet in the excel file. Useful for differentiating multiple layers in a single file. If nothing is provided will default to Sheet1, Sheet2, etc.
        errorSheetName - string of desired name for the error sheet in the excel file. Useful for differentiating multiple layers in a single file. If nothing is provided will default to Sheet1, Sheet2, etc.
    Outputs: 
        does not return anything, however the workbook provided is now filled populated with comparisons of the passed in information.
'''
def compare(workbook,caffetxt, cpptxt,caffeSheetName=None,cppSheetName=None,errorSheetName=None):

    #load text files into numpy arrays
    caffe = np.loadtxt(caffetxt)
    cpp = np.loadtxt(cpptxt)
    assert (caffe.shape == cpp.shape)
    
    #calculate error and average
    error = 1 - np.nan_to_num( np.absolute(caffe-cpp)/( np.maximum(caffe,cpp) ) )
    ave_error = np.average(error)

    #create sheets to write to
    caffeSheet = workbook.add_worksheet(caffeSheetName)
    cppSheet = workbook.add_worksheet(cppSheetName)
    errorSheet = workbook.add_worksheet(errorSheetName)

    #create fills for error sheet
    greenFill = workbook.add_format()
    greenFill.set_bg_color('green')
    redFill = workbook.add_format()
    redFill.set_bg_color('red')

    #write to caffe sheet
    if len(caffe.shape) == 2:
        for i in range(caffe.shape[0]):
            for j in range(caffe.shape[1]):
                caffeSheet.write(i+1, j, caffe[i][j])

        #write to cpp sheet
        for i in range(cpp.shape[0]):
            for j in range(cpp.shape[1]):
                cppSheet.write(i+1, j, cpp[i][j])

        #write to error sheet while coloring cells accordingly
        for i in range(error.shape[0]):
            for j in range(error.shape[1]):
                if error[i][j] > 0.85:
                    errorSheet.write(i+1, j, error[i][j],greenFill)
                else:
                    errorSheet.write(i+1, j, error[i][j],redFill)
    else:
        for i in range(caffe.shape[0]):
            caffeSheet.write(i+1, 1, caffe[i])

        #write to cpp sheet
        for i in range(cpp.shape[0]):
            cppSheet.write(i+1, 1, cpp[i])

        #write to error sheet while coloring cells accordingly
        for i in range(error.shape[0]):
            if error[i] > 0.85:
                errorSheet.write(i+1, 1, error[i],greenFill)
            else:
                errorSheet.write(i+1, 1, error[i],redFill)

    errorSheet.write('D1','Average : ')
    errorSheet.write('E1',ave_error)
 



'''  
    Description: 
        compares all files in two directories assuming filenames for each layer are stored in there respective locations in the same order and creates an excel sheet showcasing the similarities
    Inputs:
        excelFileName - name of excel file to be created
        caffe_path - path to directory holding files of caffe layer's outputs. Assumes there is a file in the directory named filenames.txt that contains a list of all other files in the directory in order.
        cpp_path  - path to directory holding files of cpp layer's outputs. Assumes there is a file in the directory named filenames.txt that contains a list of all other files in the directory in order.
    Outputs: 
        does not return anything but creates an excel file that holds all the comparisons    
'''
#given a filename to save comparisons in and path to two directories containing layer outputs for caffe and cpp networks, compares them and keeps comparisons in xlsxfile.
def auto_compare(excelFileName,caffe_path,cpp_path):

    #open files and read in both lists of files to compare
    caffe_f = open(caffe_path + 'filenames.txt','r')
    cpp_f = open(cpp_path + 'filenames.txt','r')
    caffe_lines = caffe_f.readlines()
    cpp_lines = cpp_f.readlines()

    #ensure you have the same number of files to compare in both arrays
    assert len(caffe_lines) == len(cpp_lines)
    num_files = len(caffe_lines)

    #create excel file
    workbook = xlsxwriter.Workbook(excelFileName)

    #compare each file by calling compare function
    for i in range(num_files):
        print('beginning next file')
        caffeFileName = caffe_lines[i].replace('\n','')
        cppFileName = cpp_lines[i].replace('\n','')

        caffePath = caffe_path + caffeFileName
        cppPath = cpp_path + cppFileName

        compare(workbook, caffePath, cppPath, caffeSheetName='caffe_'+caffeFileName[:-4], cppSheetName='vivado_'+cppFileName[:-4], errorSheetName='error ' + str(i))
        print(caffeFileName + ' finished')

    workbook.close()
 



'''
    Description: 
        takes a net after forward pass has occured and writes the blobs, weights and biases to files. The weights and biases are saved as arrays in cpp format.
    Inputs:
        net - the caffe net that has all the information after a forward pass has taken place.
        blobs_directory - directory to save blob files
        weights_directory - directory to save weight files
        bias_directory - directory to save bias files
        dimensions - string, either '3d' or '2d' determines whether 4 dimensional weights will be save as 2 dimensional arrays or 1 large 3 dimensional array
    Outputs:
        does not return anything but weights biases and blobs are stored in files in their respective directory.
'''
# should write weights and biases in cpp format 
def write_caffe_files(net, blobs_directory, weights_directory, bias_directory,dimensions='3d'):
    

    #open file to keep track of filenames
    f = open(blobs_directory + 'blob_filenames.txt', 'w')


    #--------------------------------------------------------------- save blobs
    #iterate through all blobs
    for blob in net.blobs:

        #save current blobs filename to make reading files later easier
        f.write(blob + '_blob.txt\n')

        #get current blobs data
        W = net.blobs[blob].data[...]

        #open file to store current blobs data
        blobs_path = blobs_directory + blob + '.txt'
        with open(blobs_path, "w") as out_file:

            #iterate through all layers that output size 4 (e.g. conv, norm, etc.)
            if(len(W.shape) == 4):

                #iterate through blob and create strings to write to file in 2D form
                for i in range(len(W)):
                    for j in range(len(W[i])):
                        for k in range(len(W[i][j])):
                            outstring = "  "
                            for l in range(len(W[i][j][k])):
                                outstring += "{:.8E}".format(Decimal(str(W[i][j][k][l]))) #
                                outstring += "  "
                            outstring += "\n"
                            out_file.write(outstring)
            #iterate through all layers that output size 2 (e.g. fc)
            elif (len(W.shape) == 2):

                #iterate through blob and create strings to write to file in 2D form
                for i in range(len(W[0])):
                    outstring = "  "
                    for j in range(len(W)):
                        outstring += "{:.8E}".format(Decimal(str(W[j][i]))) #
                        outstring += "  "
                    outstring += "\n"
                    out_file.write(outstring)

            #reports when done with entire blob before file is closed
            print('done with ', blob,' blob')
    
    #closes file containing each blobs txt file name
    f.close()
    #'''

    #open new files to hold names of weights and bias files
    f_w = open(weights_directory + 'weights_filenames.txt', 'w')
    f_b = open(bias_directory + 'bias_filenames.txt', 'w')

    #iterate through each layer
    for params in net.params:

        #save current layers filename to make reading files later easier
        f_w.write(params + '_weights.txt\n')
        f_b.write(params + '_bias.txt\n')

        #get current layers data
        W = net.params[params][0].data[...]
        B = net.params[params][1].data[...]

        ############################################################################################################################ WEIGHTS
        #save weights in 3 dimensional arrays (c++ format) if selected
        if dimensions == '3d':

            #open file to store current filenames for each weight
            weights_path = weights_directory + params + '_weights.txt'
            with open(weights_path, "w") as out_file:

                if(len(W.shape) == 4):  #iterate through all layers that have 4 dimensional weights 4 (e.g. conv)

                    #iterate through blob and create strings to write to file in 3D form
                    outstring = "{"
                    for i in range(len(W)):
                        for j in range(len(W[i])):
                            outstring += "{"
                            for k in range(len(W[i][j])):
                                outstring += "{"
                                outstring += "{:.8E}".format(Decimal(str(W[i][j][k][0]))) #
                                for l in range(1,len(W[i][j][k])):
                                    outstring += ", "
                                    outstring += "{:.8E}".format(Decimal(str(W[i][j][k][l]))) #
                                outstring += "},\n"
                            outstring = outstring[:-2]
                            outstring += "},\n\n"
                    outstring = outstring[:-3]
                    outstring += "};"
                    out_file.write(outstring)
                elif (len(W.shape) == 2):   #iterate through all layers that have 2 dimensional weights (e.g. fc)
                    #TODO: ADD comments
                    length = len(W)
                    curr_end = 1024
                    curr = 0
                    while curr < length:

                        if curr_end > length:
                            curr_end = length

                        count = 0
                        fcFile = params + '_weights_('+ str(curr) + '-' + str(curr_end) +').txt'
                        f_curr = open(weights_directory + fcFile,'w')
                        outstring = "{"
                        for i in range(curr,curr_end):
                            outstring += "{"
                            outstring += "{:.8E}".format(Decimal(str(W[i][0]))) 
                            count=1
                            for j in range(1,len(W[i])):
                                outstring += ", "
                                if count == 8:
                                    outstring+='\n'
                                    count = 0
                                outstring += "{:.8E}".format(Decimal(str(W[i][j]))) #

                                count+=1
                            outstring += "},\n"
                        outstring = outstring[:-2]
                        outstring += "};\n\n"

                        f_curr.write(outstring)
                        out_file.write(fcFile + '\n')

                        curr = curr_end
                        curr_end += 1024

        #save weights in 2 dimensional arrays (c++ format) if selected
        elif dimensions == '2d':
            
            #open file to store current filenames for each weight
            weights_path = weights_directory + params + '_weights.txt'
            with open(weights_path, "w") as out_file:

                if(len(W.shape) == 4):  #iterate through all layers that have 4 dimensional weights 4 (e.g. conv)

                    #iterate through blob and create strings to write to file in 2D form
                    for i in range(len(W)):
                        for j in range(len(W[i])):
                            outstring = "kernel {}/{} channel {}/{}\n".format(i+1,len(W),j+1,len(W[i]))
                            out_file.write(outstring)
                            outstring = "{"
                            for k in range(len(W[i][j])):
                                outstring += "{"
                                outstring += "{:.8E}".format(Decimal(str(W[i][j][k][0]))) #
                                for l in range(1,len(W[i][j][k])):
                                    outstring += ", "
                                    outstring += "{:.8E}".format(Decimal(str(W[i][j][k][l]))) #
                                outstring += "},\n"
                            outstring = outstring[:-2]
                            outstring += "};\n\n"
                            out_file.write(outstring)
                elif (len(W.shape) == 2):   #iterate through all layers that have 2 dimensional weights (e.g. fc)
                    
                    length = len(W)
                    curr_end = 1024
                    curr = 0
                    while curr < length:

                        if curr_end > length:
                            curr_end = length

                        count = 0
                        fcFile = params + '_weights_('+ str(curr) + '-' + str(curr_end) +').txt'
                        f_curr = open(weights_directory + fcFile,'w')
                        outstring = "{"
                        for i in range(curr,curr_end):
                            outstring += "{"
                            outstring += "{:.8E}".format(Decimal(str(W[i][0]))) 
                            count=1
                            for j in range(1,len(W[i])):
                                outstring += ", "
                                if count == 8:
                                    outstring+='\n'
                                    count = 0
                                outstring += "{:.8E}".format(Decimal(str(W[i][j]))) #

                                count+=1
                            outstring += "},\n"
                        outstring = outstring[:-2]
                        outstring += "};\n\n"

                        f_curr.write(outstring)
                        out_file.write(fcFile + '\n')

                        curr = curr_end
                        curr_end += 1024

        #invalid specification for weights dimension format
        else:
            print('error with weight dimensions')
        
        #reports when done with entire layer's weights
        print('done with ', params,' weight')  
            
        #open file to store current filenames for each bias
        bias_path = bias_directory + params + '_bias.txt'
        with open(bias_path, "w") as out_file:

            #iterate through current layer's bias and create string to write to file
            outstring = "{ "
            outstring += "{:.8E}".format(Decimal(str(B[0])))
            for i in range(1,len(B)):
                outstring += ",\n "
                outstring += "{:.8E}".format(Decimal(str(B[i])))
            outstring += ' };'
            out_file.write(outstring)
        
            #report when completed with current layer's bias
            print('done with ', params,' bias')  
        #'''

    #close files that were used to store filenames for weights and biases
    f_w.close()
    f_b.close()  
 



'''
    Description: 
        converts input file to a common 2D format for comparisons.
    Inputs:
        oldFile - file containing the original data tensor that is to be converted
        newFile - name of the new file that will contain the data in 2D format
        numCols - number of columns per row. When comparing, you need the same shape.
    Outputs:
        returns nothing but a file containing the data in 2D format has been created.
'''
def convert_to_2d(oldFile,newFile,numCols):
    f = open(oldFile,'r')
    content = f.read()
    content = content.replace('[','')
    content = content.replace(']','')
    content = content.replace(',','')
    content = content.replace('\n',' ')
    nums = content.split()

    f1 = open(newFile,'w')
    for i in range(1,len(nums)+1):
        f1.write( '{:20}'.format(nums[i-1] + ''))
        if i % numCols == 0:
            f1.write('\n')

    f1.close()
    f.close()
 



'''
    Description: 
        merges files. Used to merge outputs of layers that have been seperated across seperate files
    Inputs:
        fileList - a list containing all the filenames to merge
        newFile - name of the to be created containing the merged data
    Outputs:
        does not return anything but a file is created containing all the data from the list of files
'''
def mergeFiles(fileList,newFile):
    f = open(newFile,'w')

    for i in range(len(fileList)):
        f1 = open(fileList[i],'r')
        f.write( f1.read() )

    f.close()
    f1.close()












if __name__ == '__main__':


    #convert from c++ array format to 2d arrays
    #       FIRST PARAMETER IS NAME/LOCATION OF OLD FILE
    #       SECOND PARAMETER IS NAME/LOCATION OF NEW FILE THAT WILL BE CREATED
    #       THIRD PARAMETER IS NUMBER OF COLUMNS TO HAVE
    '''                 #format
    convert_to_2d('provided_caffe_files/cifar10_conv1.txt','caffe_files/conv1.txt',31)
    convert_to_2d('provided_caffe_files/cifar10_conv2.txt','caffe_files/conv2.txt',31)
    convert_to_2d('provided_caffe_files/cifar10_conv3.txt','caffe_files/conv3.txt',31)
    convert_to_2d('provided_caffe_files/cifar10_pool1.txt','caffe_files/pool1.txt',31)
    convert_to_2d('provided_caffe_files/cifar10_pool2.txt','caffe_files/pool2.txt',31)
    convert_to_2d('provided_caffe_files/cifar10_pool3.txt','caffe_files/pool3.txt',31)
    convert_to_2d('provided_caffe_files/cifar10_ip1.txt','caffe_files/fc1.txt',1)
    #'''


    #merge vivado files into one big file
    #       FIRST PARAMETER IS A LIST OF ALL FILES TO MERGE (NAME AND PATH)
    #       SECOND PARAMETER IS THE NAME/PATH TO THE NEW FILE THAT WILL BE CREATED
    '''
    conv_1_list = []
    conv_2_list = []
    conv_3_list = []
    pool_1_list = []
    pool_2_list = []
    for i in range(1,6):
        conv_1_list.append( 'Cifar10_Blob_Vivado/conv1out' + str(i) + '.txt' )
    for i in range(1,11):
        conv_2_list.append( 'Cifar10_Blob_Vivado/conv2out' + str(i) + '.txt' )
    for i in range(1,16):
        conv_3_list.append( 'Cifar10_Blob_Vivado/conv3out' + str(i) + '.txt' )
    for i in range(1,6):
        pool_1_list.append( 'Cifar10_Blob_Vivado/pool1out' + str(i) + '.txt' )
    for i in range(1,11):
        pool_2_list.append( 'Cifar10_Blob_Vivado/pool2out' + str(i) + '.txt' )

    mergeFiles(conv_1_list,'vivado_files/conv1.txt')
    mergeFiles(conv_2_list,'vivado_files/conv2.txt')
    mergeFiles(conv_3_list,'vivado_files/conv3.txt')
    mergeFiles(pool_1_list,'vivado_files/pool1.txt')
    mergeFiles(pool_2_list,'vivado_files/pool2.txt')
    #'''


    #compare
    #       FIRST PARAMETER IS NAME OF EXCEL FILE
    #       SECOND PARAMETER IS FOLDER WHERE CAFFE FILES ARE LOCATED. THERE MUST BE A FILE CALLED filenames.txt THAT CONTAINS ALL THE FILES TO BE COMPARED IN THE SAME ORDER AS THE C++ filenames.txt
    #       THIRD PARAMETER IS FOLDER WHERE C++ FILES ARE LOCATED. THERE MUST BE A FILE CALLED filenames.txt THAT CONTAINS ALL THE FILES TO BE COMPARED IN IN THE SAME ORDER AS THE C++ filenames.txt
    '''
    auto_compare('synchronous_map.xlsx','caffe_files/','vivado_files/')
    #'''

    print('\n\n\ndone')



    


