# MATH 4-shot examples from "Many-Shot In-Context Learning" paper (arXiv 2404.11018, Figure A.6)
# Copied verbatim from LightningRL/sample/sample.py for eval parity.
MATH_4SHOT_EXAMPLES = """<|im_start|>user
Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$<|im_end|>
<|im_start|>assistant
I need to find the domain of this expression.
This expression has two square root expressions.
I know the values under each square root must be non-negative.
So let me start by ensuring that.
I set $x-2 \\geq 0$, which simplifies to $x \\geq 2$.
Then, I set $5-x \\geq 0$, which simplifies to $x \\leq 5$.
But I need to make sure the denominator is not zero.
So, $5-x$ must also be greater than zero, which gives me $x < 5$.
Therefore, the domain of the expression is $[2,5)$ $\\boxed{[2,5)}$
Answer: $[2,5)$
Final Answer: The final answer is $[2,5)$. I hope it is correct.<|im_end|>
<|im_start|>user
Problem: If $\\det\\mathbf{A}=2$ and $\\det\\mathbf{B}=12$, then find $\\det(\\mathbf{A}\\mathbf{B})$.<|im_end|>
<|im_start|>assistant
I need to find the determinant of the product of matrices A and B.
I remember that the determinant of the product of two matrices equals the product of their determinants.
So, $\\det(\\mathbf{A}\\mathbf{B})=(\\det\\mathbf{A})(\\det\\mathbf{B})=(2)(12)=\\boxed{24}$
Answer: 24
Final Answer: The final answer is 24. I hope it is correct.<|im_end|>
<|im_start|>user
Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?<|im_end|>
<|im_start|>assistant
Okay, so Terrell lifts a total of $2 \\cdot 12 \\cdot 20 = 480$ pounds of weight with the 20-pound weights.
Well, if he switches to 15-pound weights, the total weight lifted will be $2 \\cdot 15 \\cdot n = 30n$ pounds, where n is the number of lifts.
I want to find the number of lifts, n, for the total weight lifted to be the same.
I equate $30n$ to 480 pounds and solve for n.
$30n = 480$
$\\Rightarrow n = 480/30 = \\boxed{16}$
Answer: 16
Final Answer: The final answer is 16. I hope it is correct.<|im_end|>
<|im_start|>user
Problem: If the system of equations $6x-4y=a$, $6y-9x=b$ has a solution $(x,y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b}$, assuming $b$ is nonzero.<|im_end|>
<|im_start|>assistant
I'm given a system of two equations.
I see that if I multiply the first equation by $-\\frac{3}{2}$, I'll get another equation that has the same left-hand side as the second equation, $6y-9x$.
Let me try that
$6y-9x=-\\frac{3}{2}a$.
Ah, I also know that $6y-9x=b$, so I can equate these two equations.
So, $-\\frac{3}{2}a=b \\Rightarrow \\frac{a}{b}=\\boxed{-\\frac{2}{3}}$.
Answer: $-\\frac{2}{3}$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.<|im_end|>
<|im_start|>user
"""
