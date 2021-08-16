import torch
import torch.optim as optim
import torch.multiprocessing as mp
import time

def chief(args, traffic_light, counter, num_of_workers,
          shared_model, shared_grad_buffers, optimizer,
          shared_obs_stats):
    num_iteration = 0
    while True:
        time.sleep(1)
        if counter.get() >= num_of_workers:
            for n, p in shared_model.named_parameters():
                p._grad = shared_grad_buffers.grads[n + '_grad']

            optimizer.step()
            counter.reset()
            shared_grad_buffers.reset()
            traffic_light.switch()  # workers start new loss computation

            num_iteration += 1
        # if num_iteration % (args.ac_update_step * 10) == 0:
        #     save_path = 'result/' + args.env_name + '_model.pt'
        #     torch.save([shared_model.state_dict(), shared_obs_stats.get_results()], save_path)
