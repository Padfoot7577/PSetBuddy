"""
This method takes a 2*2 matrix (nested lists).
the last column of the list represent all the classes each person takes
step 1: loop through each person's classes, add it in the set
step 2: make the set into a list, sort the list
step 3: check each person's classes against the master list, create vector
with 1s being yes have taken, 0s being no, have not taken
the output should be a list of lists, the lists nested within are each persons's individual class vector 
"""
#test case 1 represents four people and last element representing their current classes
test_case_1=[[[2],[0],['5.111','7.012','CC201']],
             [[15],[1],['7.012','15.401','6.006','6.036']],
             [[3],[1],["3.091","2.001","21M.900"]],
             [[6],[0],["6.01","6.031","6.006","1.02"]]]

                                                                                                

def generate_class_vectors(input_l):
    total_classes=set()
    for index in range(len(input_l)):
        for each_class in input_l[index][-1]:
            total_classes.add(each_class)
    total_classes_list=list(total_classes)
    total_classes_list.sort()
    everyone_classes=[]
    for person in input_l:
        classes=person[-1]
        class_vector=[0]*len(total_classes_list)
        for c in classes:
            ind=total_classes_list.index(c)
            class_vector[ind]=1
        everyone_classes.append(class_vector)
    return everyone_classes 

print(generate_class_vectors(test_case_1))




        

             
