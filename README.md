# IRIS
IRIS - IGT Reservoir for Image Systems

### What's an IGT device?

In normal computing, we use field effect transistors (FET) to process information, the fundamental characteristic of this type of component is the the fast response to current, changing to 0 or 1 in picoseconds.

IGT refers to **Ionic-Gated Transistor**, is a new kind of device that has a flip time of micro-seconds. The improvement is the temporal component that can be used, simulating the organic synapses in human brain.

This transistor is not based in the field effect, an ionic solution (b) is used and the ion movement is the current controller. An electric doble layer (EDL) is generated when a voltage V is applied in the gate (a), forming a Super-Capacitor (EDLC) that modulate the resistivity of the semi-conductor (SC) (c). In resting state, the SC behave as a resistor with almost infinite resistance, but, with the EDL, that resistance drops to a finite resistance, allowing current flow, called Excitatory Postsynaptic Current (EPSC) if is a positive stimulus.

This delay between the conducting and insulation state give this tool an short-memory property. This amazing capacity give this device the advantage of being capable to create compact circuits that have the same computing power with low energy cost.

### What's is reservoir computing (RC)?

The modern AI's use the layer logic for information processing. This gives extremely good results, but, it has some problems. When an input is give to the AI, the layers have little to none delay to information transport, which comes from current speed and other noises from the ambient, being bad for temporal dependent input. 

Reservoir computing is a new method, based on Recurrent Neural Networks (RNN), that is specific to this kind of situation. A reservoir is a fully connected neural layer that lets an input propagate in the processing layer, the reservoir. But, in this logic, is extremely difficult to train the reservoir weight, because it's time dependant and recursive. The answer to that problem is quite simple: don't train it.

We can think that each neuron in the reservoir layer has already processed the input, but have so much noise that can't be read. But, if all neurons have processed the information, we have a linear combination of that signal that kill the noise and results only in the answer that we want.

The training in this type of model is about fitting the output weight to match the expected answer. The reservoir computing mimics the way that biological neuron nodes works, with this intricante and fully connected layer that have redundance and temporal communication, it's not a full replica, but it's a working approximation.

### What we have done

This project aim's for two things: trying to replicate the results of the letter [Reservoir computing for image processing based on ion-gated flexible organic transistors with nonlinear synaptic dynamics](https://www.sciencedirect.com/science/article/pii/S1566119925000059) and creating our own reservoir with multiple neurons.

#### Recreation:

The article was not very clear about the nuances of their methods. In particular, the mathematical model of the IGT was not thoroughly explained, so some adaptations had to be made to approximate the reported results. (more on that in the report).

The study used the MNIST dataset to train a neural network (NN) for image classification. The objective was to downscale the input images and evaluate which approach achieved better classification performance.

Their model consisted of using a neuromorphic device, the IGT, to downscale the original image. Then, the resulting image was passed to a NN to classify the image. The goal was to compare the accuracy of a NN under different input types, in this case, with and without the IGT in the downscale process. 

Their results showed that the IGT-based preprocessing preserved information from the original pixels, obtaining an image with richer features and a better retain of the digit shape.

#### Our model:

Building upon the baseline proposed in the article, our group explored with two new approaches: a new type of downscale with the IGT and using reservoir computing to receive the treated image.

The new method of downscale aimed to use the advantages of a neuromorphic device to retain the form of the image. To achieve that, the pulse train logic was changed and rearranged to take pixel blocks of 2x2. This approach produced promising results, allowing the downscaled images to retain the overall shape of the original digits.

One challenge associated with this type of downscale is the introduction of noise. To address this issue, the group used a **gaussian filter** and two IGT devices in the treatment. The resulting images were then subtracted from one another to receive a clear downscaled image.

The second approach was to test the memory properties of the reservoir computing, sending the image as a temporal pulse sequence. This method requires only a single solution of a least-squares problem during training and seeks to emulate certain aspects of biological information processing observed in the brain.

