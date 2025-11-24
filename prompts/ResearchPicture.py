prompt = r"""
You are an expert ML illustrator.
Draw a clean, NeurIPS/ICLR-style scientific figure using Nano Banana Pro.

GOAL:
Create a professional, publication-quality diagram that exactly follows the
structure and logic provided in the MODULE LIST below.
Do not invent components, do not reinterpret, do not add creativity.
Strictly follow the logical flow.

GLOBAL RULES:

- Flat, clean NeurIPS style (no gradients, no gloss, no shadows)
- Consistent thin line weights
- Professional pastel palette
- Rounded rectangles for blocks
- Arrows must clearly indicate data flow
- No long sentences, only short labels
- Keep spacing clean and balanced
- All modules must appear exactly once unless specified

LAYOUT:

- Horizontal left → right layout (recommended)
- Or vertical top → bottom if modules are inherently sequential
- Align components cleanly in straight lines
- Respect the module order exactly as listed

MODULE LIST (FILL THIS WITH YOUR PAPER'S CONTENT):

1. Input(s):
   - [Your input items]

2. Preprocessing / Encoding / Embedding:
   - [Your modules]

3. Core Architecture / Stages / Blocks:
   - [Your modules in exact order]

4. Special Mechanisms (optional):
   - [Attention / memory / routing / dynamic paths]

5. Output Head:
   - [Your output block]

NOTES (Optional but useful):

- Specify any required two-branch or multi-branch flow
- Specify “A and B must merge here”
- Specify “keep this as a single tall block with submodules”
- If experimental plot → replace section above with structured numbers

STYLE REQUIREMENTS:

- NeurIPS 2024 visual tone
- Very light background
- Text left-aligned inside blocks
- Arrows short and clean
- Use consistent vertical spacing

Generate the final diagram.
"""