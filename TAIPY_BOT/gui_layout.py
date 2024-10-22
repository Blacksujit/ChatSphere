# GUI Layout
page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# Taipy **Chat**{: .color-primary}
<|New Conversation|button|class_name=fullwidth plain|on_action=reset_chat|>
### Previous activities
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|>
<|"Generating..."|text|class_name=generating-text|visible={is_generating}|>
|>
|>
|>
"""