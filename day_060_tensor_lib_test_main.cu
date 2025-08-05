#include "day_060_tensor_lib.cu"

int main()
{
    printf("=== Tensor Library Examples ===\n\n");

    // Example 1: Create 1D tensor and print
    printf("1. Creating 1D tensor: tensor T(1.0f, 2.0f, 3.0f, 4.0f)\n");
    tensor T(1.0f, 2.0f, 3.0f, 4.0f);
    printf("Result: ");
    T.Print();
    printf("\n");

    // Example 2: Reshape to 2D
    printf("2. Reshaping to 2x2: T.Reshape(2, 2)\n");
    tensor T2 = T.Reshape(2, 2);
    printf("Result: ");
    T2.Print();
    printf("\n");

    // Example 3: Create larger tensor
    printf("3. Creating larger tensor: tensor T3(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)\n");
    tensor T3(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    printf("Result: ");
    T3.Print();
    printf("\n");

    // Example 4: Reshape to 2x3
    printf("4. Reshaping to 2x3: T3.Reshape(2, 3)\n");
    tensor T4 = T3.Reshape(2, 3);
    printf("Result: ");
    T4.Print();
    printf("\n");

    // Example 5: Create new tensor and reshape to 3x2
    printf("5. Creating new tensor and reshaping to 3x2\n");
    tensor T5Base(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    tensor T5 = T5Base.Reshape(3, 2);
    printf("Result: ");
    T5.Print();
    printf("\n");

    // Example 6: Single element tensor
    printf("6. Single element tensor: tensor T6(42.0f)\n");
    tensor T6(42.0f);
    printf("Result: ");
    T6.Print();
    printf("\n");

    // Example 7: Larger 1D tensor (8 elements)
    printf("7. Larger 1D tensor: tensor T7(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f)\n");
    tensor T7(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    printf("Result: ");
    T7.Print();
    printf("\n");

    // Example 8: Reshape to 4x2
    printf("8. Reshaping to 4x2: T7.Reshape(4, 2)\n");
    tensor T8 = T7.Reshape(4, 2);
    printf("Result: ");
    T8.Print();
    printf("\n");

    printf("=== All examples completed successfully! ===\n");

    printf("\n=== Element-wise Operators Testing ===\n\n");

    // Test 1: Binary tensor addition
    printf("1. Tensor + Tensor: [1.0, 2.0, 3.0, 4.0] + [0.5, 1.5, 2.5, 3.5]\n");
    tensor A1(1.0f, 2.0f, 3.0f, 4.0f);
    tensor B1(0.5f, 1.5f, 2.5f, 3.5f);
    tensor C1 = A1 + B1;
    printf("Result: ");
    C1.Print();
    printf("\n");

    // Test 2: Binary tensor subtraction
    printf("2. Tensor - Tensor: [5.0, 6.0, 7.0, 8.0] - [1.0, 2.0, 3.0, 4.0]\n");
    tensor A2(5.0f, 6.0f, 7.0f, 8.0f);
    tensor B2(1.0f, 2.0f, 3.0f, 4.0f);
    tensor C2 = A2 - B2;
    printf("Result: ");
    C2.Print();
    printf("\n");

    // Test 3: Binary tensor multiplication
    printf("3. Tensor * Tensor: [2.0, 3.0, 4.0, 5.0] * [1.0, 2.0, 3.0, 4.0]\n");
    tensor A3(2.0f, 3.0f, 4.0f, 5.0f);
    tensor B3(1.0f, 2.0f, 3.0f, 4.0f);
    tensor C3 = A3 * B3;
    printf("Result: ");
    C3.Print();
    printf("\n");

    // Test 4: Binary tensor division
    printf("4. Tensor / Tensor: [8.0, 12.0, 18.0, 24.0] / [2.0, 3.0, 6.0, 8.0]\n");
    tensor A4(8.0f, 12.0f, 18.0f, 24.0f);
    tensor B4(2.0f, 3.0f, 6.0f, 8.0f);
    tensor C4 = A4 / B4;
    printf("Result: ");
    C4.Print();
    printf("\n");

    // Test 5: Tensor + scalar
    printf("5. Tensor + Scalar: [1.0, 2.0, 3.0, 4.0] + 5.0\n");
    tensor A5(1.0f, 2.0f, 3.0f, 4.0f);
    tensor C5 = A5 + 5.0f;
    printf("Result: ");
    C5.Print();
    printf("\n");

    // Test 6: Scalar + tensor
    printf("6. Scalar + Tensor: 10.0 + [1.0, 2.0, 3.0, 4.0]\n");
    tensor A6(1.0f, 2.0f, 3.0f, 4.0f);
    tensor C6 = 10.0f + A6;
    printf("Result: ");
    C6.Print();
    printf("\n");

    // Test 7: Tensor - scalar
    printf("7. Tensor - Scalar: [10.0, 20.0, 30.0, 40.0] - 5.0\n");
    tensor A7(10.0f, 20.0f, 30.0f, 40.0f);
    tensor C7 = A7 - 5.0f;
    printf("Result: ");
    C7.Print();
    printf("\n");

    // Test 8: Scalar - tensor
    printf("8. Scalar - Tensor: 50.0 - [10.0, 20.0, 30.0, 40.0]\n");
    tensor A8(10.0f, 20.0f, 30.0f, 40.0f);
    tensor C8 = 50.0f - A8;
    printf("Result: ");
    C8.Print();
    printf("\n");

    // Test 9: Tensor * scalar
    printf("9. Tensor * Scalar: [1.0, 2.0, 3.0, 4.0] * 3.0\n");
    tensor A9(1.0f, 2.0f, 3.0f, 4.0f);
    tensor C9 = A9 * 3.0f;
    printf("Result: ");
    C9.Print();
    printf("\n");

    // Test 10: Tensor / scalar
    printf("10. Tensor / Scalar: [6.0, 12.0, 18.0, 24.0] / 3.0\n");
    tensor A10(6.0f, 12.0f, 18.0f, 24.0f);
    tensor C10 = A10 / 3.0f;
    printf("Result: ");
    C10.Print();
    printf("\n");

    // Test 11: In-place tensor += tensor
    printf("11. In-place +=: [1.0, 2.0, 3.0, 4.0] += [2.0, 3.0, 4.0, 5.0]\n");
    tensor A11(1.0f, 2.0f, 3.0f, 4.0f);
    tensor B11(2.0f, 3.0f, 4.0f, 5.0f);
    A11 += B11;
    printf("Result: ");
    A11.Print();
    printf("\n");

    // Test 12: In-place tensor *= scalar
    printf("12. In-place *= scalar: [2.0, 4.0, 6.0, 8.0] *= 2.5\n");
    tensor A12(2.0f, 4.0f, 6.0f, 8.0f);
    A12 *= 2.5f;
    printf("Result: ");
    A12.Print();
    printf("\n");

    // Test 13: 2D tensor operations
    printf("13. 2D Tensor operations: [[1.0, 2.0], [3.0, 4.0]] + [[0.5, 1.5], [2.5, 3.5]]\n");
    tensor A13(1.0f, 2.0f, 3.0f, 4.0f);
    tensor A132D = A13.Reshape(2, 2);
    tensor B13(0.5f, 1.5f, 2.5f, 3.5f);
    tensor B132D = B13.Reshape(2, 2);
    tensor C13 = A132D + B132D;
    printf("Result: ");
    C13.Print();
    printf("\n");

    printf("=== All operator tests completed successfully! ===\n");

    printf("\n=== Matrix Multiplication Testing ===\n\n");

    // Test 1: Basic 2x2 matrix multiplication
    printf("1. Basic 2x2 MatMul: [[1.0, 2.0], [3.0, 4.0]] @ [[2.0, 0.0], [1.0, 2.0]]\n");
    tensor M1(1.0f, 2.0f, 3.0f, 4.0f);
    tensor M12D = M1.Reshape(2, 2);
    tensor M2(2.0f, 0.0f, 1.0f, 2.0f);
    tensor M22D = M2.Reshape(2, 2);
    tensor MResult1 = M12D.MatMul(M22D);
    printf("Result: ");
    MResult1.Print();
    printf("Expected: [[4.0, 4.0], [10.0, 8.0]]\n\n");

    // Test 2: Rectangular matrix multiplication (2x3) × (3x2)
    printf("2. Rectangular MatMul: (2x3) × (3x2)\n");
    printf("A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n");
    printf("B = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]\n");
    tensor ARect(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    tensor A2x3 = ARect.Reshape(2, 3);
    tensor BRect(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    tensor B3x2 = BRect.Reshape(3, 2);
    tensor MResult2 = A2x3.MatMul(B3x2);
    printf("Result: ");
    MResult2.Print();
    printf("Expected: [[22.0, 28.0], [49.0, 64.0]]\n\n");

    // Test 3: Vector multiplication (dot product) (1x4) × (4x1)
    printf("3. Vector dot product: (1x4) × (4x1)\n");
    printf("A = [[1.0, 2.0, 3.0, 4.0]]\n");
    printf("B = [[2.0], [3.0], [4.0], [5.0]]\n");
    tensor V1(1.0f, 2.0f, 3.0f, 4.0f);
    tensor V11x4 = V1.Reshape(1, 4);
    tensor V2(2.0f, 3.0f, 4.0f, 5.0f);
    tensor V24x1 = V2.Reshape(4, 1);
    tensor MResult3 = V11x4.MatMul(V24x1);
    printf("Result: ");
    MResult3.Print();
    printf("Expected: [[40.0]] (1*2 + 2*3 + 3*4 + 4*5 = 40)\n\n");

    // Test 4: Outer product (4x1) × (1x3)
    printf("4. Outer product: (4x1) × (1x3)\n");
    printf("A = [[1.0], [2.0], [3.0], [4.0]]\n");
    printf("B = [[2.0, 3.0, 4.0]]\n");
    tensor O1(1.0f, 2.0f, 3.0f, 4.0f);
    tensor O14x1 = O1.Reshape(4, 1);
    tensor O2(2.0f, 3.0f, 4.0f);
    tensor O21x3 = O2.Reshape(1, 3);
    tensor MResult4 = O14x1.MatMul(O21x3);
    printf("Result: ");
    MResult4.Print();
    printf("Expected: [[2.0, 3.0, 4.0], [4.0, 6.0, 8.0], [6.0, 9.0, 12.0], [8.0, 12.0, 16.0]]\n\n");

    // Test 5: Identity matrix multiplication
    printf("5. Identity matrix test: [[1.0, 2.0], [3.0, 4.0]] @ [[1.0, 0.0], [0.0, 1.0]]\n");
    tensor I1(1.0f, 2.0f, 3.0f, 4.0f);
    tensor I12x2 = I1.Reshape(2, 2);
    tensor I2(1.0f, 0.0f, 0.0f, 1.0f); // Identity matrix
    tensor I22x2 = I2.Reshape(2, 2);
    tensor MResult5 = I12x2.MatMul(I22x2);
    printf("Result: ");
    MResult5.Print();
    printf("Expected: [[1.0, 2.0], [3.0, 4.0]] (unchanged)\n\n");

    // Test 6: Error case - incompatible dimensions
    printf("6. Error test - incompatible dimensions: (2x3) × (2x2)\n");
    tensor E1(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    tensor E12x3 = E1.Reshape(2, 3);
    tensor E2(1.0f, 2.0f, 3.0f, 4.0f);
    tensor E22x2 = E2.Reshape(2, 2);
    tensor MError = E12x3.MatMul(E22x2);
    printf("Result: ");
    MError.Print();
    printf("Expected: Error message and empty tensor []\n\n");

    printf("=== All MatMul tests completed! ===\n");
    return 0;
}
