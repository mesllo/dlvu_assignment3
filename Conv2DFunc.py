class Conv2DFunc(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward
  passes which operate on Tensors.
  """
  @staticmethod
  def forward(ctx, input_batch, kernel, stride=1, padding=1):
    """
    In the forward pass we receive a Tensor containing the input
    and return a Tensor containing the output. ctx is a context
    object that can be used to stash information for backward
    computation. You can cache arbitrary objects for use in the
    backward pass using the ctx.save_for_backward method.
    """
    # store objects for the backward
    ctx.save_for_backward(input)
    ctx.save_for_backward(kernel)
    
    # your code here
    padded = torch.nn.ConstantPad2d(padding, 0)(input_batch)
    unfolded = torch.nn.Unfold(kernel.shape[-2:], stride=stride)(padded)
    ctx.save_for_backward(unfolded)
    muled = kernel.reshape(1, 1, -1) @ unfolded
    out_w = int((in_tensor.shape[-1] + 2*padding - k.shape[1])/stride + 1)
    out_h = int((in_tensor.shape[-2] + 2*padding - k.shape[0])/stride + 1)
    output_batch = torch.nn.functional.fold(muled, (out_w, out_h), 1)
    return output_batch
  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass we receive a Tensor containing the
    gradient of the loss with respect to the output, and we need
    to compute the gradient of the loss with respect to the
    input
    """
    # retrieve stored objects
    input, kernel, unfolded = ctx.saved_tensors
    # your code here
    input_batch_grad = None
    kernel_grad = unfolded @ grad_output.reshape(input.shape[0], -1, 1)
    return input_batch_grad, kernel_grad, None, None