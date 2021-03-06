?	?ZH@?}@?ZH@?}@!?ZH@?}@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?ZH@?}@?y??X|s@1?(???c@A?x?n?|??I???' @*	?????#?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?????@!?Z?R;3X@)?????@1?Z?R;3X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??:ǀ???!??i'???)??:ǀ???1??i'???:Preprocessing2F
Iterator::Modelqt???!?~???	@)???9d??1??8@???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????Z???!?(??'? @)?s
򳑫?1??j?O??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?^?sa?@!
$???7X@)??u??q?1?+$?ٲ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?tj??P@Q?+?ڏ@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?y??X|s@?y??X|s@!?y??X|s@      ??!       "	?(???c@?(???c@!?(???c@*      ??!       2	?x?n?|???x?n?|??!?x?n?|??:	???' @???' @!???' @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?tj??P@y?+?ڏ@@?"7
model_6/conv2d_316/Conv2DConv2DN?????!N?????0"l
Bgradient_tape/model_6/batch_normalization_316/FusedBatchNormGradV3FusedBatchNormGradV3	????!??!)5
HH??"7
model_6/conv2d_316/BiasAddBiasAddEx8??S??!2?\?2??"h
<gradient_tape/model_6/conv2d_320/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?8??G???!P?S??0"V
0model_6/batch_normalization_316/FusedBatchNormV3FusedBatchNormV3??L?0??!?SxFι?"7
model_6/conv2d_318/Conv2DConv2D3?:?d??!꫿ٱ???0"7
model_6/conv2d_328/Conv2DConv2D
????!K} <?P??0"7
model_6/conv2d_322/Conv2DConv2DQ???f??!???????0"7
model_6/conv2d_325/Conv2DConv2D???H??!???	?h??0"f
;gradient_tape/model_6/conv2d_320/Conv2D/Conv2DBackpropInputConv2DBackpropInput????ԅ?!P????0Q      Y@Y?va?????a&z6 ??X@q??˪?I@yT???0q?"?

both?Your program is POTENTIALLY input-bound because 65.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?51.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 