# Blog Outline: JEPA-demo

## Working Title

JEPA-demo: low-effort obstacle understanding with I-JEPA representations

## Thesis

I-JEPA is not a detector, but its self-supervised visual representations may still provide useful low-effort signals for operational AI: what kind of obstacle is present and where it roughly appears.

## Outline

1. Why operational AI needs cheap visual understanding probes
2. What I-JEPA provides: representation learning, not detection
3. Dataset: YOLO-format obstacle labels as benchmark reference
4. "What is it?": nearest labeled examples in I-JEPA embedding space
5. "Where is it?": patch-level saliency from I-JEPA features
6. Visual comparison: green YOLO boxes vs red I-JEPA estimate
7. Failure modes and what they teach us
8. Next steps: trained lightweight head, segmentation priors, or fine-tuning

## Practical Framing

The value is not claiming I-JEPA replaces YOLO. The value is seeing whether a representation model can quickly produce useful approximate signals before investing in a full supervised detector.
