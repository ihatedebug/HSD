#include"fpga_api.h"
#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/mman.h>
#include<cstring>
//#include<stdlib.h>

#define min(x,y) (((x)<(y))?(x):(y))
#define ABS(s) ((s)>0 ? (s) : -(s))

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = m_size;
  v_size_ = v_size;

  m1_size_ = v_size * v_size;

  data_size_ = (m_size_+1)*v_size_; // fpga bram data size
  data_size_M = (2*v_size_)*v_size_*sizeof(float);

  fd_ = open("/dev/mem", O_RDWR);
  data_M = static_cast<float*>(mmap(NULL, data_size_M, PROT_READ|PROT_WRITE, MAP_SHARED, fd_, data_addr));
  data_ = new float[data_size_];	

  output_ = static_cast<unsigned int*>(mmap(NULL, sizeof(unsigned int), PROT_READ|PROT_WRITE, MAP_SHARED,fd_, output_addr));
  output_MV = new unsigned int[m_size_];
  // output_M = static_cast<unsigned int*>(NULL);

  num_block_call_ = 0;
}

FPGA::~FPGA()
{
  munmap(data_M, data_size_M);
  munmap(output_, sizeof(unsigned int));
  close(fd_);

  delete[] data_;
}

float* FPGA::matrix(void)
{
  return data_ + v_size_;
}

float* FPGA::vector(void)
{
  return data_;
}

float* FPGA::matrix_M1(void)
{
  return data_M;
}

float* FPGA::matrix_M2(void)
{
  return data_M + m1_size_;
}

void FPGA::reset(void)
{
  num_block_call_ = 0;
}

int FPGA::num_block_call(void)
{
  return num_block_call_;
}

const float* FPGA::blockMV()
{
  num_block_call_ += 1;

  // cpu version
  float* vec = this->vector();
  float* mat = this->matrix();
  float* out  = reinterpret_cast<float*>(output_MV);  

  for(int i = 0; i < m_size_; ++i)
  {
    out[i] = 0;
    for(int j = 0; j < v_size_; ++j)
      out[i] += vec[j] * mat[v_size_*i + j];
  }

  for(int i = 0; i < m_size_; ++i)
    data_[i] = out[i];

  return data_;    
}

const float* __attribute__((optimize("O0"))) FPGA::blockMM()
{
  num_block_call_ += 1;

  // fpga version
  *output_ = 0x5555;
  while(*output_ == 0x5555);

  return data_M;    
}

void FPGA::largeMV(const float *large_mat, const float *input, float *output, int num_input, int num_output)
{
    float *vec = this->vector();
    float *mat = this->matrix();

    // 0) Initialize output vector
    for (int i = 0; i < num_output; ++i)
        output[i] = 0;

    for (int i = 0; i < num_output; i += m_size_)
    {
        for (int j = 0; j < num_input; j += v_size_)
        {
            // 0) Initialize input vector
            int block_row = min(m_size_, num_output - i);
            int block_col = min(v_size_, num_input - j);

            // 1) Assign a vector
            memcpy(vec, input + j, sizeof(float) * block_col);
            // 2) Assign a matrix
            memset(mat, 0, sizeof(float) * m_size_ * v_size_);
            for (int k = 0; k < block_row; k++)
            {
                memcpy(mat + v_size_ * k, large_mat + i * num_input + k * num_input + j, sizeof(float) * block_col);
            }
            // 3) Call a function `blockMV() to execute MV multiplication
            const float *ret = this->blockMV();

            // 4) Accumulate intermediate results
            for (int row = 0; row < block_row; ++row)
                output[i + row] += ret[row];
        }
    }
}

void FPGA::largeMM(const float *weight_mat, const float *input_mat, float *output, int num_input, int num_output, int num_matrix2)
{
    float *m1 = this->matrix_M1();
    float *m2 = this->matrix_M2();
    
    #define GROUP_NUM (num_output/4)
    #define LEFTOVER (num_output%4)
    #define ELEM_NUM 4

    int nonzero_row_num = ((GROUP_NUM)*2 + LEFTOVER);
    float *nonzero_data = new float[nonzero_row_num*num_input];
    int *nonzero_rows = new int[nonzero_row_num*num_input];

    // 0) Initialize output vector
    for (int i = 0; i < num_output * num_matrix2; ++i)
        output[i] = 0;

    for (int j = 0; j < num_output; j++){
        // 1) Compute Non-zero data & indices
        for(int i = 0; i< GROUP_NUM ; i++){
            float test_block [ELEM_NUM] = {ABS(weight_mat[j*num_input + i*ELEM_NUM]), ABS(weight_mat[j*num_input + i*ELEM_NUM + 1]), ABS(weight_mat[j*num_input + i*ELEM_NUM + 2]), ABS(weight_mat[j*num_input + i*ELEM_NUM + 3])};
            int min1_index = 0;
            int min2_index = 1;
            if(test_block[2]<test_block[min1_index])
                min1_index = 2;
            else if (test_block[2]<test_block[min2_index])
                min2_index = 2;
            if(test_block[3]<test_block[min1_index])
                min1_index = 3;
            else if (test_block[3]<test_block[min2_index])
                min2_index = 3;
            if(min1_index>min2_index){
                int tmp = min2_index;
                min2_index = min1_index;
                min1_index = tmp;
            }
            nonzero_data[nonzero_row_num*j+ i*2] = weight_mat[j*num_input + i*ELEM_NUM + min1_index];
            nonzero_data[nonzero_row_num*j+ i*2 + 1] = weight_mat[j*num_input + i*ELEM_NUM + min2_index];
            nonzero_rows[nonzero_row_num*j + i*2] = i + i*ELEM_NUM + min1_index;
            nonzero_rows[nonzero_row_num*j + i*2 + 1] = i*ELEM_NUM + min2_index;
        }
        for (int i = 0; i < LEFTOVER ; i++){
            nonzero_data[(GROUP_NUM) * 2 + i] = weight_mat[j*num_input + (GROUP_NUM) * ELEM_NUM + i];
            nonzero_rows[(GROUP_NUM) * 2 + i] = GROUP_NUM * ELEM_NUM + i ;
        }
    }

    for (int i = 0; i < nonzero_row_num; i += v_size_)
    {
        for (int j = 0; j < num_input; j += v_size_)
        {
            for (int k = 0; k < num_matrix2; k += v_size_)
            {
                // 0) Initialize input vector
                int block_row = min(v_size_, num_output - i);
                int block_col_1 = min(v_size_/2, num_input - j);
                int block_col_2 = min(v_size_, num_matrix2 - k);

                // 1) Assign a m1
                memset(m1, 0, sizeof(float) * v_size_/2 * v_size_); // row * col_1
                for (int i1 = 0; i1 < block_row; i1++)
                    memcpy(m1 + i1 * v_size_/2, weight_mat + (i + i1) * num_input + j, sizeof(float) * block_col_1);
                // 2) Assign a m2
                memset(m2, 0, sizeof(float) * v_size_ * v_size_); // col_1 * col_2
                for (int j2 = 0; j2 < num_output; j2++){
                    for(int i2 = 0; i2 < nonzero_row_num; i2++){
                        memcpy(m2 + i2 * v_size_, input_mat + (j + i2) * num_matrix2 + nonzero_rows[nonzero_row_num * j2 + i2], sizeof(float) * block_col_2);   
                    }
                }
                
                // 3) Call a function `blockMM() to execute Matrix matrix multiplication
                const float *ret = this->blockMM();

                // 4) Accumulate intermediate results
                for (int n = 0; n < block_row; ++n)
                {
                    for (int m = 0; m < block_col_2; ++m)
                    {
                        output[(i + n) + (k + m) * num_output] += ret[n * v_size_ + m];
                    }
                }
            }
        }
    }
}

void FPGA::convLowering(const std::vector<std::vector<std::vector<std::vector<float>>>> &cnn_weights,
                        std::vector<std::vector<float>> &new_weights,
                        const std::vector<std::vector<std::vector<float>>> &inputs,
                        std::vector<std::vector<float>> &new_inputs)
{
    /*
     * Arguments:
     *
     * conv_weights: [conv_channel, input_channel, conv_height, conv_width]
     * new_weights: [?, ?]
     * inputs: [input_channel, input_height, input_width]
     * new_inputs: [?, ?]
     *
     */

    int conv_channel = cnn_weights.size();
    int input_channel = cnn_weights[0].size();
    int conv_height = cnn_weights[0][0].size();
    int conv_width = cnn_weights[0][0][0].size();
    // int input_channel = cnn_weights.size();
    int input_height = inputs[0].size();
    int input_width = inputs[0][0].size();

    // IMPLEMENT THIS
    // For example,
    // new_weights[0][0] = cnn_weights[0][0][0][0];
    // new_inputs[0][0] = inputs[0][0][0];
    // weights
    for (int i = 0; i < conv_channel; i++)
    {
        for (int j = 0; j < input_channel; j++)
        {
            for (int k = 0; k < conv_height; k++)
            {
                for (int l = 0; l < conv_width; l++)
                {
                    new_weights[i][l + k * conv_width + j * conv_height * conv_width] = cnn_weights[i][j][k][l];
                }
            }
        }
    }
    // input
    int row_ops = input_height - conv_height + 1;
    int column_ops = input_width - conv_width + 1;
    for (int i = 0; i < input_channel; i++)
    {
        for (int j = 0; j < row_ops ; j++)
        {
            for (int k = 0; k < column_ops ; k++)
            {
                for (int l = 0; l < conv_height; l++)
                {
                    for (int m = 0; m < conv_width; m++)
                    {
                        // XXX: input_height - conv_height = input_width - conv_width always?
                        new_inputs[i * conv_height * conv_width + l * conv_width + m][j * column_ops + k] = inputs[i][j + l][k + m];
                    }
                    
                }
            }
        }
    }
}