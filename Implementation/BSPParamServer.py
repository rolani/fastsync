import time
import sys
import ray
import random
import Server1


###########################################################################
# Synchronous Parameter Server Training
# -------------------------------------
# We'll now create a synchronous parameter server training scheme. We'll first
# instantiate a process for the parameter server, along with multiple
# workers.
num_workers = int(sys.argv[1])
iterations = 1000
out_file = "BSP_out{0}_acc_loss.txt".format(num_workers)
time_file = "BSP_out{0}_iter_time.txt".format(num_workers)

f_out = open(out_file, "a")
f_time = open(time_file, "a")

ray.init(ignore_reinit_error=True, _redis_password="password")
ps = Server1.ParameterServer.remote(1e-2)
workers = [Server1.Worker.remote() for i in range(num_workers)]
###########################################################################
# We'll also instantiate a model on the driver process to evaluate the test
# accuracy during training.

model = Server1.MobileNetV2()
test_loader = Server1.get_data_loader()[1]
#train_loader = Server.get_data_loader_cinic_net()[0]

###########################################################################
# Training alternates between:
#
# 1. Computing the gradients given the current weights from the server
# 2. Updating the parameter server's weights with the gradients.
print("Running Synchronous Parameter Server Training.")
start_time = time.time()
init_start_time = time.time()
current_weights = ps.get_weights.remote()

accuracy = 0
i = 0
while accuracy <= 60:
    begin_time = time.time()
    gradients = [
        worker.compute_gradients.remote(current_weights) for worker in workers
    ]
    print("Iteration {0}".format(i))
    # Calculate update after all gradients are available.
    current_weights = ps.apply_gradients.remote(*gradients)

    # print("Iteration time is: ".format(time.time() - begin_time))
    # f_time.write('{:.5f}'.format(time.time() - begin_time))
    if i % 2000 == 0:
        f_time.write("{} {:.5f}".format(i, time.time() - start_time))
        # Evaluate the current model.
        model.set_weights(ray.get(current_weights))
        loss, accuracy = Server1.evaluate(model, test_loader)
        print("Iter {}: \taccuracy is {:.3f} \tloss is {:.6f}".format(i, accuracy, loss))
        #end_time = time.time()
        f_out.write('{} {:3f} {:6f}\n'.format(i, accuracy, loss))
        f_time.write(" {:.3f}\n".format(accuracy))
        start_time = time.time()
    i += 1

#model.set_weights(ray.get(current_weights))
#loss, accuracy = Server1.evaluate(model, test_loader)
#print("Iter {}: \taccuracy is {:.3f} \tloss is {:.6f}".format(i, accuracy, loss))
end_time = time.time()
f_time.write("{:.5f}".format(time.time() - init_start_time))
print("Final Runtime is: {0}".format(time.time() - init_start_time))
print("Final accuracy is {:.3f}.".format(accuracy))
print("Final loss is {:.6f}.".format(loss))
# Clean up Ray resources and processes before the next example.
ray.shutdown()
print("Distributed BSP training completed with {0} workers".format(num_workers))
f_out.close()
f_time.close()
