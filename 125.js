/* webpack --config  webpack.dll.config.js --progress */

const path = require('path');
const webpack = require('webpack');
const lib = require('./build/lib.dependencies');

module.exports = {
  entry: {
    lib: lib
  },
  output: {
    path: path.join(__dirname, 'lib'),
    filename: '[name].js',
    /**
     * output.library
     * 将会定义为 window.${output.library}
     * 在这次的例子中，将会定义为`window.vendor_library`
     */
    library: '[name]'
  },
  plugins: [
    new webpack.DllPlugin({
      /**
       * path
       * 定义 manifest 文件生成的位置
       * [name]的部分由entry的名字替换
       */
      path: path.join(__dirname, 'lib', 'manifest.json'),
      /**
       * name
       * dll bundle 输出到那个全局变量上
       * 和 output.library 一样即可。
       */
      name: '[name]',
      context: __dirname
    }),
    new webpack.optimize.OccurenceOrderPlugin(),
    new webpack.optimize.UglifyJsPlugin({
      mangle: {
        except: ['$', 'exports', 'require']
      },
      // exclude:/\.min\.js$/
      // mangle:true,
      compress: { warnings: false },
      output: { comments: false }
    })
  ]
};
