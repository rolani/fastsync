import time
import Server1
import ray
import random
import sys


###########################################################################
# Asynchronous Parameter Server1 Training
# --------------------------------------
# We'll now create a synchronous parameter Server1 training scheme. We'll first
# instantiate a process for the parameter Server1, along with multiple
# workers.

iterations = 50
num_workers = int(sys.argv[1])
#print("{}".format(Server1.num_workers))

out_file = "ASP_out{0}_acc_loss.txt".format(num_workers)
time_file = "ASP_out{0}_iter_time.txt".format(num_workers)

f_out = open(out_file, "a")
f_time = open(time_file, "a")

print("Running Asynchronous Parameter Server1 Training.")

ray.init(ignore_reinit_error=True)
ps = Server1.ParameterServer.remote(1e-2)
workers = [Server1.Worker.remote() for i in range(num_workers)]

###########################################################################
# Here, workers will asynchronously compute the gradients given its
# current weights and send these gradients to the parameter Server1 as
# soon as they are ready. When the Parameter Server1 finishes applying the
# new gradient, the Server1 will send back a copy of the current weights to the
# worker. The worker will then update the weights and repeat.

###########################################################################
# We'll also instantiate a model on the driver process to evaluate the test
# accuracy during training.

model = Server1.MobileNetV2()
test_loader = Server1.get_data_loader()[1]

###########################################################################
start_time = time.time()
init_start_time = time.time()
current_weights = ps.get_weights.remote()

# initial computation
gradients = {}
for worker in workers:
    gradients[worker.compute_gradients.remote(current_weights)] = worker

print(gradients)
#for worker in workers:
    #print("Worker {0} is on prior iteration {1}".format(worker, ray.get(worker.get_timestep.remote())))

accuracy = 0
i = 0
while accuracy <= 60:
    print("Iteration {0}".format(i))
    ready_gradient_list, _ = ray.wait(list(gradients))
    ready_gradient_id = ready_gradient_list[0]
    worker = gradients.pop(ready_gradient_id)
    #print(worker)
    #print("Async time is: {0}".format(time.time() - start_time))
    # Compute and apply gradients.
    current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
    gradients[worker.compute_gradients.remote(current_weights)] = worker
    #print(gradients)
    max_time = 0
    done = False
    for worker in workers:
        #print("Worker {0} is on iteration {1}".format(worker, ray.get(worker.get_time_step.remote())))
        if ray.get(worker.get_time_step.remote()) % 2000 == 0 and done == False:
            # Evaluate the current model.
            max_time = ray.get(worker.get_time_step.remote())
            f_time.write("{} {:.5f}".format(max_time, time.time() - start_time))
            model.set_weights(ray.get(current_weights))
            loss, accuracy = Server1.evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.3f} \tloss is {:.6f} \t Iteration {}".format(
                max_time, accuracy, loss, i))
            # end_time = time.time()
            f_out.write('{} {:3f} {:6f} {}\n'.format(max_time, accuracy, loss, i))
            f_time.write(" {:.3f}\n".format(accuracy))
            done = True
            start_time = time.time()
    i += 1

#model.set_weights(ray.get(current_weights))
#loss, accuracy = Server1.evaluate(model, test_loader)
#print("Iter {}: \taccuracy is {:.3f} \tloss is {:.6f}".format(i, accuracy, loss))
end_time = time.time()
f_time.write("{:.5f}".format(time.time() - init_start_time))
print("Final Runtime is: {0}".format(time.time() - init_start_time))#f_time.write("{:.5f}".format(time.time() - start_time))
print("Final accuracy is {:.3f}.".format(accuracy))
print("Final loss is {:.6f}.".format(loss))
ray.shutdown()
print("Distributed ASP training completed with {0} workers".format(num_workers))
f_out.close()
f_time.close()
