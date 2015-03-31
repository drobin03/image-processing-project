require 'sinatra/asset_pipeline'
require 'sinatra/base'
require 'bootstrap-sass'
require 'sass'
require 'coffee-script'
require 'compass'

class Site < Sinatra::Base
  register Sinatra::AssetPipeline

  get '/' do
    images = Dir.entries('./public/images/').select { |f| File.fnmatch('*.jpg', f) }
    erb :index, locals: { images: images }
  end

  post '/solve' do
    image_name = params[:image].sub('.jpg', '')
    puzzle = `python main.py public/images/#{image_name}.jpg`
  end

  get '/results' do
    image_name = params[:image].sub('.jpg', '')
    erb :results, locals: { image_name: image_name }, layout: false
  end


  template :results do
    '<img src="/images/<%= image_name %>/Solved.jpg" />'
  end

  # get '/js/solver.js' do
  #   coffee :solver
  # end

  # get '/css/app.css' do
  #   scss :app, :style => :expanded
  # end

  # start the server if ruby file executed directly
  run! if app_file == $0
end
# post '/upload' do
#   name =  params[:image].original_filename
#   directory = "/"
#   # create the file path
#   path = File.join(directory, name)
#   # write the file
#   File.open(path, "wb") { |f| f.write(upload['datafile'].read) }