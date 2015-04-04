$ ->
  $('#upload_form').submit (e) ->
    e.stopPropagation()
    console.log("uploaded")