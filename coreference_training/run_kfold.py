import subprocess,os
import winsound         # for sound  
import time             # for sleep

def run_test():
    CREATE_NO_WINDOW = 0x08000000
    log_folder = input("which log to start from? ")
    des_input = input("Enter a description for the files: ")
    log_s = int(log_folder)
    L1_sizes = [200]
    for log_number,L1_size in zip(range(log_s,log_s+len(L1_sizes)),L1_sizes):
        
        print("Running log ",log_number, " with L1_size ", L1_size)
        file_path = "output_log\log%02d"%log_number
        
        if not os.path.exists(file_path):
            print("Creating path " + file_path)
            os.makedirs(file_path)
        else:
            print("Path exist " + file_path)
        
        loss = []
        accuracy = []
        final_lines = []            
        for offset in range(0,10):
            
            file_name = "\offset%02d.txt"%offset
            filename = file_path+file_name
            
            with open(filename, "w") as output:
                print("Exec offset ",offset)
                process = subprocess.Popen(["python","train.py","--offset", str(offset), \
                                            "--L1_size",str(L1_size)],stdout=output, \
                                           creationflags=CREATE_NO_WINDOW)
                process.wait()
            print("done with "+filename+"\n")
            
            with open(filename,"r") as f:
                lines = f.readlines()
        
                for line in reversed(lines):
                    if line.startswith("Step"):
                        print(line)
                        final_lines.append(line)
                        x = line.replace("%","").replace(",","").split()
                        loss.append(float(x[3]))
                        accuracy.append(int(x[5]))
                        break;
                    
        print("-----------")
        avg_loss = sum(loss)/len(loss)
        avg_acc = sum(accuracy)/len(accuracy)
        accuracy.append(avg_acc)
        loss.append(avg_loss)
        final_result = "result of the k fold, avg_accuracy: " + str(avg_acc) + \
              " loss: " + str(avg_loss)
        print(final_result)

        
        with open(file_path +"\descrip.txt","w") as f:
            f.write("# " + des_input +"\n")
            if L1_size > 0 :
                f.write("#using 1 hidden layer with " + str(L1_size) + " units\n")
            else:
                f.write("#No hidden layer\n")
            for line in final_lines:
                f.write(line)
            f.write(final_result)
    winsound.Beep(800, 800) # frequency, duration
    time.sleep(0.25)        # in seconds (0.25 is 250ms)
       
def read_files(log_s,log_e):
    all_loss = []
    all_acc = []
    all_des = []
    for log_number in range(log_s,log_e+1):
        log_id = "log%02d"%log_number
        loss = [log_id]
        accuracy = [log_id]
        all_des.append(log_id)
        file_path = "output_log\\" + log_id
        with open(file_path +"\descrip.txt","r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("#"):
                    all_des.append(line)
                else:
                    if line.startswith("Step"):
                        x = line.replace("%","").replace(",","").split()
                        loss.append(float(x[3]))
                        accuracy.append(int(x[5]))
        avg_loss = sum(loss[1:])/len(loss[1:])
        avg_acc = sum(accuracy[1:])/len(accuracy[1:])
        accuracy.append(avg_acc)
        loss.append(avg_loss)                
        all_acc.append(accuracy)
        all_loss.append(loss)
    for a in all_acc:
        print(a)
    for l in all_loss:
        print(l)
    for line in all_des:
        print(line)
